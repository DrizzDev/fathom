from __future__ import annotations

from logging import getLogger
from typing import Any, Dict, List, Optional, cast

from fathom.constants.completion import GateOutcome
from fathom.constants.observability import CompletionEvent
from fathom.constants.state import CommonStateKey, IntentStateKey, PlanMetadataKey
from fathom.core.agent.completion import CompletionGate
from fathom.core.services.criterion import CriterionObserver
from fathom.schemas.completion import CompletionEvidence, GateDecision
from fathom.schemas.criterion import CriterionDecision
from fathom.schemas.effect import ActionEffect, ActionEffectStatus
from fathom.schemas.observability import CompletionLogContext
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

    Multi-signal architecture mirroring the main branch's proven policy.
    Each turn produces a typed CompletionEvidence bundle (claim, action,
    screen, optional criterion) which the CompletionGate adjudicates per
    sub-goal kind:

      - ACTION sub-goals require asserted claim AND justified rationale AND
        a dispatched action that caused screen evolution. Equivalent to
        main's 3-of-3 with the screen-verified gate on action_executed.
      - VALIDATION sub-goals short-circuit on an asserted claim; otherwise
        require any two of justified rationale and screen-verified dispatch.

    The CriterionObserver remains as an additive RCA-grade signal. Its
    verdict is folded into CompletionEvidence.criterion and logged on every
    decision, but it never vetoes an otherwise-conclusive gate outcome.
    """

    def __init__(
        self,
        *,
        context: GraphContext,
        criterion_observer: CriterionObserver,
        gate: Optional[CompletionGate] = None,
    ) -> None:
        """
        Bind the evaluator to its graph context, criterion observer, and gate.
        """

        self.__context = context
        self.__criterion_observer = criterion_observer
        self.__gate = gate if gate is not None else CompletionGate()

    async def evaluate(
        self,
        *,
        plan: Any,
        step_result: StepResult,
        accumulated: List[StepResult],
        observation: Optional[ScreenObservation] = None,
    ) -> Optional[IntentGraphState]:
        """
        Assess this turn's evidence and either advance the sub-goal or retain it.
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

        active = cast("SubGoal", current)

        criterion_decision = await self.__observe_criterion(
            active=active,
            observation=observation,
            step_result=step_result,
        )

        last_effect = agent_state.get_last_action_effect()

        evidence = self.__context.reasoner.assess_completion(
            sub_goal=active,
            analysis=analysis,
            effect=last_effect,
            criterion_decision=criterion_decision,
            delta_score=agent_state.last_delta_score,
            screen_changed=step_result.screen_changed,
            screen_description=step_result.observation or step_result.step.action.target or "",
        )
        self.__log_evidence_assessed(
            active=active,
            evidence=evidence,
            effect=last_effect,
            step_result=step_result,
        )

        decision = self.__gate.adjudicate(evidence=evidence, sub_goal=active)
        self.__log_gate_adjudicated(
            active=active,
            evidence=evidence,
            decision=decision,
            effect=last_effect,
            step_result=step_result,
        )

        if decision.outcome is GateOutcome.ADVANCE:
            signal = self.__build_storage_signal(
                active=active,
                analysis=analysis,
                step_result=step_result,
            )
            return self.__advance_or_complete(
                current=active,
                signal=signal,
                evidence=evidence,
                accumulated=accumulated,
                kind=action_kind_for(step_result.step.action.action_type),
            )

        self.__log_retained(
            active=active,
            evidence=evidence,
            decision=decision,
            step_result=step_result,
        )
        return None

    async def __observe_criterion(
        self,
        *,
        active: SubGoal,
        step_result: StepResult,
        observation: Optional[ScreenObservation],
    ) -> Optional[CriterionDecision]:
        """
        Run the criterion observer for RCA telemetry; never used to gate.
        """

        if observation is None:
            return None

        decision = await self.__criterion_observer.check(
            sub_goal=active,
            observation=observation,
            workflow_id=self.__context.workflow_id,
        )
        logger.info(
            "Criterion observer reported verdict",
            extra={
                **self.__log_context(),
                "sub_goal.index": active.index,
                "sub_goal.kind": active.kind.value,
                "criterion.source": decision.source.value,
                "criterion.verdict": decision.verdict.value,
                "criterion.confidence": decision.confidence,
                "criterion.evidence": list(decision.evidence),
                "sub_goal.description": active.description[:80],
                "step.screen_changed": step_result.screen_changed,
                "event": CompletionEvent.CRITERION_OBSERVED.value,
            },
        )
        return decision

    def __build_storage_signal(
        self,
        *,
        active: SubGoal,
        step_result: StepResult,
        analysis: AnalysisResult,
    ) -> SubGoalCompletionSignal:
        """
        Compute the legacy SubGoalCompletionSignal used by mark_current_sub_goal_complete.
        """

        return self.__context.reasoner.analyze_subgoal_completion(
            analysis=analysis,
            sub_goal_description=active.description,
            screen_changed=step_result.screen_changed,
            delta_score=self.__context.agent_state.last_delta_score,
            screen_description=step_result.observation or step_result.step.action.target or "",
        )

    @staticmethod
    def __is_evaluable(*, current: Optional[SubGoal], has_sub_goals: bool) -> bool:
        """
        Whether a sub-goal context exists for evaluation this turn.
        """

        return current is not None and has_sub_goals

    def __advance_or_complete(
        self,
        *,
        current: SubGoal,
        kind: ActionKind,
        evidence: CompletionEvidence,
        accumulated: List[StepResult],
        signal: SubGoalCompletionSignal,
    ) -> IntentGraphState:
        """
        Mark the current sub-goal complete; route to next sub-goal or VERIFY.
        """

        agent_state = self.__context.agent_state
        has_more = agent_state.mark_current_sub_goal_complete(completion_signal=signal)

        if has_more:
            return self.__retry_for_next_sub_goal(
                current=current, evidence=evidence, accumulated=accumulated, kind=kind
            )

        return self.__route_to_verify(
            current=current, evidence=evidence, accumulated=accumulated, kind=kind
        )

    def __retry_for_next_sub_goal(
        self,
        *,
        current: SubGoal,
        kind: ActionKind,
        evidence: CompletionEvidence,
        accumulated: List[StepResult],
    ) -> IntentGraphState:
        """
        Emit a graph patch that loops back to GROUND for the next sub-goal.
        """

        agent_state = self.__context.agent_state
        next_sub_goal = agent_state.get_current_sub_goal()

        logger.info(
            "Sub-goal advanced; looping back to GROUND for next sub-goal",
            extra={
                **self.__log_context(),
                "kind": kind.value,
                "sub_goal.index": current.index,
                "sub_goal.kind": current.kind.value,
                "evidence.notes": list(evidence.notes),
                "sub_goal.description": current.description[:80],
                "event": CompletionEvent.SUBGOAL_ADVANCED.value,
                "next.sub_goal.index": next_sub_goal.index if next_sub_goal else None,
                "next.sub_goal.description": (
                    next_sub_goal.description[:80] if next_sub_goal else None
                ),
                "next.sub_goal.kind": (next_sub_goal.kind.value if next_sub_goal else None),
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
        kind: ActionKind,
        evidence: CompletionEvidence,
        accumulated: List[StepResult],
    ) -> IntentGraphState:
        """
        Mark intent complete and route to VERIFY when the last sub-goal advances.
        """

        agent_state = self.__context.agent_state
        completion_reason = "All sub-goals completed sequentially"

        agent_state.mark_complete(reason=completion_reason)

        logger.info(
            "All sub-goals advanced; routing to VERIFY for final adjudication",
            extra={
                **self.__log_context(),
                "kind": kind.value,
                "sub_goal.index": current.index,
                "sub_goal.kind": current.kind.value,
                "evidence.notes": list(evidence.notes),
                "event": CompletionEvent.INTENT_COMPLETED.value,
                "sub_goal.description": current.description[:80],
            },
        )
        return cast(
            "IntentGraphState",
            {
                CommonStateKey.IS_COMPLETE: True,
                IntentStateKey.STEP_RESULTS: accumulated,
                CommonStateKey.COMPLETION_REASON: completion_reason,
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

    def __log_evidence_assessed(
        self,
        *,
        active: SubGoal,
        step_result: StepResult,
        evidence: CompletionEvidence,
        effect: Optional[ActionEffect],
    ) -> None:
        """
        Structured log: per-turn evidence bundle assembled by the reasoner.
        """

        logger.info(
            "Completion evidence assessed",
            extra={
                **self.__log_context(),
                "sub_goal.index": active.index,
                "sub_goal.kind": active.kind.value,
                "screen.evolved": evidence.screen.evolved,
                "claim.asserted": evidence.claim.asserted,
                "claim.justified": evidence.claim.justified,
                "action.dispatched": evidence.action.dispatched,
                "sub_goal.description": active.description[:80],
                "event": CompletionEvent.EVIDENCE_ASSESSED.value,
                "criterion.observed": (
                    evidence.criterion.observed if evidence.criterion is not None else None
                ),
                "evidence.notes": list(evidence.notes),
                "step.screen_changed": step_result.screen_changed,
                "planner.emitted_action_type": step_result.step.action.action_type.value,
                "effect.status": (effect.status.value if effect is not None else None),
                "veto.applied": self.__no_progress_vetoed(
                    effect=effect,
                    screen_evolved=evidence.screen.evolved,
                    screen_changed=step_result.screen_changed,
                ),
            },
        )

    def __log_gate_adjudicated(
        self,
        *,
        active: SubGoal,
        decision: GateDecision,
        step_result: StepResult,
        evidence: CompletionEvidence,
        effect: Optional[ActionEffect],
    ) -> None:
        """
        Structured log: completion-gate decision for this turn.
        """

        logger.info(
            "Completion gate adjudicated",
            extra={
                **self.__log_context(),
                "sub_goal.index": active.index,
                "sub_goal.kind": active.kind.value,
                "gate.outcome": decision.outcome.value,
                "screen.evolved": evidence.screen.evolved,
                "claim.asserted": evidence.claim.asserted,
                "claim.justified": evidence.claim.justified,
                "action.dispatched": evidence.action.dispatched,
                "event": CompletionEvent.GATE_ADJUDICATED.value,
                "step.screen_changed": step_result.screen_changed,
                "gate.retain_reason": (
                    decision.retain_reason.value if decision.retain_reason is not None else None
                ),
                "effect.status": (effect.status.value if effect is not None else None),
                "veto.applied": self.__no_progress_vetoed(
                    effect=effect,
                    screen_evolved=evidence.screen.evolved,
                    screen_changed=step_result.screen_changed,
                ),
            },
        )

    @staticmethod
    def __no_progress_vetoed(
        *, screen_changed: bool, screen_evolved: bool, effect: Optional[ActionEffect]
    ) -> bool:
        """
        Return True iff NO_PROGRESS overrode the high-sensitivity screen_changed signal on this turn.
        """

        if effect is None:
            return False

        return (
            effect.status is ActionEffectStatus.NO_PROGRESS
            and screen_changed
            and not screen_evolved
        )

    def __log_retained(
        self,
        *,
        active: SubGoal,
        decision: GateDecision,
        step_result: StepResult,
        evidence: CompletionEvidence,
    ) -> None:
        """
        Structured log: sub-goal retained for another planner turn.
        """

        logger.info(
            "Sub-goal retained; replanning required",
            extra={
                **self.__log_context(),
                "sub_goal.index": active.index,
                "sub_goal.kind": active.kind.value,
                "evidence.notes": list(evidence.notes),
                "sub_goal.description": active.description[:80],
                "event": CompletionEvent.SUBGOAL_RETAINED.value,
                "step.screen_changed": step_result.screen_changed,
                "planner.emitted_action_type": step_result.step.action.action_type.value,
                "gate.retain_reason": (
                    decision.retain_reason.value if decision.retain_reason is not None else None
                ),
            },
        )

    def __log_skipped(self, *, reason: str, step_result: StepResult) -> None:
        """
        Structured log: evaluation skipped (failed step, missing analysis, etc.).
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


__all__ = ["SubGoalEvaluator", "CompletionLogContext"]
