from __future__ import annotations

from logging import getLogger
from typing import Any, Dict, List, Optional, cast

from fathom.constants import ActionType
from fathom.constants.reasoning import IMPLICIT_COMPLETION_THRESHOLD
from fathom.constants.state import CommonStateKey, IntentStateKey, PlanMetadataKey
from fathom.core.services.criterion import CriterionChecker
from fathom.schemas.criterion import CriterionDecision, CriterionVerdict
from fathom.schemas.observation import ScreenObservation
from fathom.schemas.reasoning import SubGoalCompletionSignal
from fathom.schemas.results import AnalysisResult, PlanResult
from fathom.schemas.steps import StepResult
from fathom.schemas.subgoal import SubGoal
from fathom.schemas.vision import ActionKind, action_kind_for
from fathom.strategies.graph.context import GraphContext
from fathom.strategies.graph.state import IntentGraphState

logger = getLogger(__name__)


class SubGoalEvaluator:
    """
    Decide whether the executed step satisfies the active sub-goal.

    Criterion-first architecture. The decomposer assigns each sub-goal an
    observable ``criterion`` (e.g. ``"'HSR Layout' is selected as the current
    location"``). After each executed step the evaluator asks one question:
    is that criterion satisfied on the current screen?

    The :class:`CriterionChecker` returns a tri-state verdict:

    - ``SATISFIED`` — advance the sub-goal.
    - ``UNSATISFIED`` — keep the sub-goal pending. Log and let the planner
      take another turn.
    - ``UNCLEAR`` — fall through to the implicit-completion streak guard.
      The streak only fires for genuine fraud cases (planner claims done +
      screen unchanged + criterion not observably true); it never accepts a
      bare claim against an explicit ``UNSATISFIED`` verdict.

    The decomposer-emitted ``directive`` is retained as a planner hint and as
    a telemetry field on every criterion decision, but it no longer gates
    advancement. Divergence between the decomposer's planned action and the
    planner's emitted action is no longer a failure mode — only an
    unsatisfied criterion is.
    """

    def __init__(
        self,
        *,
        context: GraphContext,
        criterion_checker: CriterionChecker,
    ) -> None:
        """
        Bind the evaluator to its shared graph context and criterion checker.
        """

        self.__context = context
        self.__criterion_checker = criterion_checker

    async def evaluate(
        self,
        *,
        plan: Any,
        step_result: StepResult,
        accumulated: List[StepResult],
        observation: Optional[ScreenObservation] = None,
    ) -> Optional[IntentGraphState]:
        """
        Top-level dispatch: criterion-first, streak as safety net.
        """

        agent_state = self.__context.agent_state
        current = agent_state.get_current_sub_goal()

        if not self.__is_evaluable(current=current, has_sub_goals=agent_state.has_sub_goals()):
            return None

        if not step_result.success:
            self.__log_skipped(reason="step.failed", step_result=step_result)
            return None

        analysis = self.__analysis_from(plan=plan)
        if analysis is None:
            return None

        emitted = step_result.step.action.action_type
        active = cast("SubGoal", current)
        signal = self.__context.reasoner.analyze_subgoal_completion(
            analysis=analysis,
            sub_goal_description=active.description,
            delta_score=agent_state.last_delta_score,
            screen_changed=step_result.screen_changed,
            screen_description=step_result.observation or step_result.step.action.target or "",
        )

        # No typed observation means the OBSERVE node didn't write one this
        # turn (capture failure, etc.); we cannot do a criterion check and
        # must fall back to the streak path so the agent loop keeps a guard.
        if observation is None:
            return self.__evaluate_via_streak(
                current=active,
                emitted=emitted,
                step_result=step_result,
                signal=signal,
                accumulated=accumulated,
                decision=None,
                reason="observation.missing",
            )

        decision = await self.__criterion_checker.check(
            workflow_id=self.__context.workflow_id,
            sub_goal=active,
            observation=observation,
        )

        if decision.verdict is CriterionVerdict.SATISFIED:
            return self.__advance_on_criterion(
                current=active,
                emitted=emitted,
                step_result=step_result,
                signal=signal,
                accumulated=accumulated,
                decision=decision,
            )

        if decision.verdict is CriterionVerdict.UNSATISFIED:
            self.__log_criterion_unsatisfied(
                current=active,
                emitted=emitted,
                step_result=step_result,
                decision=decision,
            )
            active.completion_claim_streak = 0
            return None

        # UNCLEAR — fall through to the implicit-completion safety net.
        return self.__evaluate_via_streak(
            current=active,
            emitted=emitted,
            step_result=step_result,
            signal=signal,
            accumulated=accumulated,
            decision=decision,
            reason="criterion.unclear",
        )

    def __advance_on_criterion(
        self,
        *,
        current: SubGoal,
        emitted: ActionType,
        step_result: StepResult,
        signal: SubGoalCompletionSignal,
        accumulated: List[StepResult],
        decision: CriterionDecision,
    ) -> IntentGraphState:
        """
        Criterion satisfied → reset streak and advance.
        """

        current.completion_claim_streak = 0
        self.__log_criterion_satisfied(
            current=current,
            emitted=emitted,
            step_result=step_result,
            decision=decision,
        )
        return self.__advance_or_complete(
            current=current,
            signal=signal,
            accumulated=accumulated,
            kind=action_kind_for(emitted),
        )

    def __evaluate_via_streak(
        self,
        *,
        current: SubGoal,
        emitted: ActionType,
        step_result: StepResult,
        signal: SubGoalCompletionSignal,
        accumulated: List[StepResult],
        decision: Optional[CriterionDecision],
        reason: str,
    ) -> Optional[IntentGraphState]:
        """
        Implicit-completion safety net for UNCLEAR (or missing) criterion verdicts.

        This is the only remaining streak path. It exists to handle two cases
        the criterion checker cannot resolve:

        - Capture failure: no typed observation reached the evaluator.
        - Genuinely ambiguous criteria (behavioural state with no observable
          post-state token) where the LLM check returns UNCLEAR.

        The streak fires only when the planner emits a completion-shaped action
        (``validate`` or ``complete``) with ``flagged_complete`` true. A single
        claim is rejected to keep the original fraud guard intact; a second
        consecutive claim is accepted to unstick a genuinely unresolvable sub-goal.
        """

        if not (self.__is_completion_emit(emitted=emitted) and signal.flagged_complete):
            current.completion_claim_streak = 0
            self.__log_criterion_unclear(
                current=current,
                emitted=emitted,
                step_result=step_result,
                decision=decision,
                reason=reason,
            )
            return None

        current.completion_claim_streak += 1

        if current.completion_claim_streak < IMPLICIT_COMPLETION_THRESHOLD:
            self.__log_criterion_unclear(
                current=current,
                emitted=emitted,
                step_result=step_result,
                decision=decision,
                reason=reason,
            )
            return None

        return self.__advance_on_implicit_completion(
            current=current,
            emitted=emitted,
            signal=signal,
            accumulated=accumulated,
            decision=decision,
            reason=reason,
        )

    def __advance_on_implicit_completion(
        self,
        *,
        current: SubGoal,
        emitted: ActionType,
        signal: SubGoalCompletionSignal,
        accumulated: List[StepResult],
        decision: Optional[CriterionDecision],
        reason: str,
    ) -> IntentGraphState:
        """
        Accept a sustained completion claim as implicit advancement.
        """

        logger.info(
            "Accepting implicit completion: sustained claim against UNCLEAR criterion",
            extra={
                **self.__log_context(),
                "event": "subgoal.implicit.completion",
                "sub_goal.index": current.index,
                "sub_goal.description": current.description[:80],
                "sub_goal.directive": self.__directive_value(current=current),
                "planner.emitted_action_type": emitted.value,
                "sub_goal.completion_claim_streak": current.completion_claim_streak,
                "signal.flagged_complete": signal.flagged_complete,
                "criterion.verdict": (decision.verdict.value if decision is not None else None),
                "criterion.source": (decision.source.value if decision is not None else None),
                "criterion.fallback_reason": reason,
            },
        )
        current.completion_claim_streak = 0
        return self.__advance_or_complete(
            current=current,
            signal=signal,
            accumulated=accumulated,
            kind=action_kind_for(emitted),
        )

    @staticmethod
    def __is_completion_emit(*, emitted: ActionType) -> bool:
        """
        Whether the planner emit is one of the "task done" action types.
        """

        return emitted in (ActionType.VALIDATE, ActionType.COMPLETE)

    @staticmethod
    def __is_evaluable(*, current: Optional[SubGoal], has_sub_goals: bool) -> bool:
        """
        Whether a sub-goal context exists for evaluation.
        """

        return current is not None and has_sub_goals

    def __advance_or_complete(
        self,
        *,
        current: SubGoal,
        signal: SubGoalCompletionSignal,
        accumulated: List[StepResult],
        kind: ActionKind,
    ) -> IntentGraphState:
        """
        Mark the current sub-goal complete; route to GROUND retry or VERIFY.
        """

        agent_state = self.__context.agent_state
        has_more = agent_state.mark_current_sub_goal_complete(completion_signal=signal)

        if has_more:
            return self.__retry_for_next_sub_goal(
                current=current, signal=signal, accumulated=accumulated, kind=kind
            )

        return self.__route_to_verify(
            current=current, signal=signal, accumulated=accumulated, kind=kind
        )

    def __retry_for_next_sub_goal(
        self,
        *,
        current: SubGoal,
        signal: SubGoalCompletionSignal,
        accumulated: List[StepResult],
        kind: ActionKind,
    ) -> IntentGraphState:
        """
        Emit a graph patch that loops back to GROUND for the next sub-goal.
        """

        agent_state = self.__context.agent_state
        next_sub_goal = agent_state.get_current_sub_goal()
        logger.info(
            "Sub-goal advanced locally; looping back to GROUND for next sub-goal",
            extra={
                **self.__log_context(),
                "kind": kind.value,
                "event": "subgoal.advanced",
                "signal.count": signal.count_signals(),
                "sub_goal.index": current.index,
                "sub_goal.description": current.description[:80],
                "sub_goal.directive": self.__directive_value(current=current),
                "next.sub_goal.index": next_sub_goal.index if next_sub_goal else None,
                "next.sub_goal.description": (
                    next_sub_goal.description[:80] if next_sub_goal else None
                ),
                "next.sub_goal.directive": self.__directive_value(current=next_sub_goal),
            },
        )
        return cast(
            "IntentGraphState",
            {
                IntentStateKey.SHOULD_RETRY: True,
                IntentStateKey.STEP_RESULTS: accumulated,
            },
        )

    def __route_to_verify(
        self,
        *,
        current: SubGoal,
        signal: SubGoalCompletionSignal,
        accumulated: List[StepResult],
        kind: ActionKind,
    ) -> IntentGraphState:
        """
        Mark intent complete and route to VERIFY when the last sub-goal advances.
        """

        agent_state = self.__context.agent_state
        completion_reason = "All sub-goals completed sequentially"
        agent_state.mark_complete(reason=completion_reason)

        logger.info(
            "All sub-goals advanced; routing to VERIFY for final intent adjudication",
            extra={
                **self.__log_context(),
                "kind": kind.value,
                "event": "subgoal.all_complete",
                "signal.count": signal.count_signals(),
                "sub_goal.index": current.index,
                "sub_goal.description": current.description[:80],
                "sub_goal.directive": self.__directive_value(current=current),
            },
        )
        return cast(
            "IntentGraphState",
            {
                CommonStateKey.IS_COMPLETE: True,
                CommonStateKey.COMPLETION_REASON: completion_reason,
                IntentStateKey.STEP_RESULTS: accumulated,
            },
        )

    @staticmethod
    def __analysis_from(*, plan: Any) -> Optional[AnalysisResult]:
        """
        Reconstruct the :class:`AnalysisResult` attached to plan metadata.
        """

        if not isinstance(plan, PlanResult) or not plan.metadata:
            return None

        raw = plan.metadata.get(PlanMetadataKey.ANALYSIS.value)
        if raw is None:
            return None

        return raw if isinstance(raw, AnalysisResult) else AnalysisResult.model_validate(raw)

    @staticmethod
    def __directive_value(*, current: Optional[SubGoal]) -> Optional[str]:
        """
        Serialised directive value for logs (None if absent).
        """

        if current is None or current.directive is None:
            return None
        return current.directive.value

    def __log_criterion_satisfied(
        self,
        *,
        current: SubGoal,
        emitted: ActionType,
        step_result: StepResult,
        decision: CriterionDecision,
    ) -> None:
        """
        Structured log: criterion observably satisfied; sub-goal will advance.
        """

        logger.info(
            "Sub-goal criterion satisfied; advancing",
            extra={
                **self.__log_context(),
                "event": "subgoal.criterion.satisfied",
                "sub_goal.index": current.index,
                "sub_goal.description": current.description[:80],
                "sub_goal.directive": self.__directive_value(current=current),
                "planner.emitted_action_type": emitted.value,
                "criterion.verdict": decision.verdict.value,
                "criterion.source": decision.source.value,
                "criterion.confidence": decision.confidence,
                "criterion.evidence": list(decision.evidence),
                "step.screen_changed": step_result.screen_changed,
            },
        )

    def __log_criterion_unsatisfied(
        self,
        *,
        current: SubGoal,
        emitted: ActionType,
        step_result: StepResult,
        decision: CriterionDecision,
    ) -> None:
        """
        Structured log: criterion observably not satisfied; sub-goal stays pending.
        """

        logger.info(
            "Sub-goal criterion not satisfied; keeping sub-goal pending",
            extra={
                **self.__log_context(),
                "event": "subgoal.criterion.unsatisfied",
                "sub_goal.index": current.index,
                "sub_goal.description": current.description[:80],
                "sub_goal.directive": self.__directive_value(current=current),
                "planner.emitted_action_type": emitted.value,
                "criterion.verdict": decision.verdict.value,
                "criterion.source": decision.source.value,
                "criterion.confidence": decision.confidence,
                "criterion.evidence": list(decision.evidence),
                "step.screen_changed": step_result.screen_changed,
            },
        )

    def __log_criterion_unclear(
        self,
        *,
        current: SubGoal,
        emitted: ActionType,
        step_result: StepResult,
        decision: Optional[CriterionDecision],
        reason: str,
    ) -> None:
        """
        Structured log: criterion unclear or absent; falling to streak path.
        """

        logger.info(
            "Sub-goal criterion unclear; falling back to streak safety net",
            extra={
                **self.__log_context(),
                "event": "subgoal.criterion.unclear",
                "sub_goal.index": current.index,
                "sub_goal.description": current.description[:80],
                "sub_goal.directive": self.__directive_value(current=current),
                "planner.emitted_action_type": emitted.value,
                "criterion.verdict": (decision.verdict.value if decision is not None else None),
                "criterion.source": (decision.source.value if decision is not None else None),
                "criterion.confidence": decision.confidence if decision is not None else 0.0,
                "criterion.fallback_reason": reason,
                "sub_goal.completion_claim_streak": current.completion_claim_streak,
                "step.screen_changed": step_result.screen_changed,
            },
        )

    def __log_skipped(self, *, reason: str, step_result: StepResult) -> None:
        """
        Structured log: evaluation skipped (failed step, etc.).
        """

        logger.info(
            "Skipping sub-goal completion check",
            extra={
                **self.__log_context(),
                "reason": reason,
                "error.message": step_result.error,
                "event": "subgoal.evaluate.skipped",
            },
        )

    def __log_context(self) -> Dict[str, Any]:
        """
        Shared structured-logging fields for every completion-gate event.
        """

        return {
            "component": "graph.intent.completion",
            "workflow.id": self.__context.workflow_id,
        }
