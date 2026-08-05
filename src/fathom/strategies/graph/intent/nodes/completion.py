from __future__ import annotations

from logging import getLogger
from typing import Any, Dict, List, Optional, cast

from pydantic import BaseModel, ConfigDict, Field

from fathom.constants.completion import RetainReason
from fathom.constants.observability import CompletionEvent
from fathom.constants.state import (
    CommonStateKey,
    CompletionReason,
    IntentStateKey,
    VerifyMode,
)
from fathom.constants.turn.advancement import AdvanceKind, ObservationPhase
from fathom.core.agent.advancement import AdvancementPolicy
from fathom.core.agent.eligibility import Eligibility
from fathom.core.agent.stall import StallPolicy
from fathom.core.exceptions import InvariantViolation
from fathom.core.services.criterion import CriterionObserver
from fathom.schemas.advancement import Advancement
from fathom.schemas.binding import Binding
from fathom.schemas.completion import ActionEvidence, ClaimEvidence
from fathom.schemas.criterion import CriterionDecision, Verdict
from fathom.schemas.effect import EffectReading
from fathom.schemas.observability import CompletionLogContext
from fathom.schemas.observation import ScreenObservation
from fathom.schemas.results import AnalysisResult, PlanResult
from fathom.schemas.shadow import GoalCursor, ShadowTurn, ShadowTurnDraft
from fathom.schemas.steps import StepResult
from fathom.schemas.subgoal import GoalState
from fathom.schemas.success import (
    ObservationRequirement,
    ObservedSuccess,
    Success,
)
from fathom.schemas.turn import TurnEvidence
from fathom.strategies.graph.context import GraphContext
from fathom.strategies.graph.intent.nodes.shadow import ShadowRunner
from fathom.strategies.graph.state import IntentGraphState

logger = getLogger(__name__)


class ProbeResult(BaseModel):
    """
    The pre-dispatch live decision and, when it advanced, the graph transition it produced.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    advancement: Advancement = Field(description="The live pre-dispatch advancement decision.")
    transition: Optional[IntentGraphState] = Field(
        default=None, description="Graph transition when the probe advanced, else None."
    )


class SubGoalEvaluator:
    """
    Decide whether the executed step satisfies the active sub-goal's canonical success.

    Each turn assembles a typed TurnEvidence correlated to the active goal's Success
    variant and hands it to the single AdvancementPolicy:

      - ObservedSuccess advances on a fresh satisfied verdict for its own observation;
        pre-dispatch satisfaction advances as SATISFIED_PRIOR.
      - CommandSuccess advances only on a matching executed action (plus its
        postcondition when present); a visible postcondition never advances before
        the command runs.
      - CaptureSuccess advances only on the correlated executed STORE.

    Momentum is owned solely by StallPolicy: a retaining turn escalates only when the
    stall signal is STALLED.
    """

    def __init__(
        self,
        *,
        context: GraphContext,
        criterion_observer: CriterionObserver,
        policy: Optional[AdvancementPolicy] = None,
    ) -> None:
        """
        Bind the evaluator to its graph context, criterion observer, and advancement policy.
        """

        self.__context = context
        self.__criterion_observer = criterion_observer
        self.__policy = policy if policy is not None else AdvancementPolicy()
        self.__stall = StallPolicy()
        self.__runner = ShadowRunner()

    async def evaluate(
        self,
        *,
        plan: Any,
        step_result: StepResult,
        accumulated: List[StepResult],
        binding: Optional[Binding] = None,
        reading: Optional[EffectReading] = None,
        observation: Optional[ScreenObservation] = None,
    ) -> Optional[IntentGraphState]:
        """
        Assess this turn's evidence and either advance the sub-goal or retain it.
        """

        agent_state = self.__context.agent_state
        current = agent_state.get_current_sub_goal()

        if current is None or not agent_state.has_sub_goals():
            return None

        if not step_result.success:
            self.__log_skipped(reason="step.failed", step_result=step_result)
            self.__record_failed(
                plan=plan, active=current, receipt=step_result, observation=observation
            )
            return None

        analysis = self.__analysis_from(plan=plan)
        if analysis is None:
            return None

        active = current

        turn = await self.__post_dispatch_evidence(
            active=active,
            analysis=analysis,
            binding=binding,
            reading=reading,
            step_result=step_result,
            observation=observation,
        )
        advancement = self.__policy.decide(success=active.success, evidence=turn)
        self.__log_adjudicated(active=active, advancement=advancement, step_result=step_result)

        if advancement.kind in (AdvanceKind.ADVANCE, AdvanceKind.SATISFIED_PRIOR):
            patch: Optional[IntentGraphState] = self.__advance_or_complete(
                current=active, accumulated=accumulated
            )
        elif advancement.kind in (AdvanceKind.ESCALATE, AdvanceKind.UNSATISFIABLE):
            patch = self.__escalate(active=active, advancement=advancement, accumulated=accumulated)
        else:
            self.__log_retained(active=active, advancement=advancement, step_result=step_result)
            patch = None

        self.__record_executed(
            plan=plan, active=active, receipt=step_result, observation=observation, live=advancement
        )
        return patch

    async def probe(self, *, observation: Optional[ScreenObservation]) -> ProbeResult:
        """
        Return the real pre-dispatch live decision for the active goal, advancing an already-satisfied
        ObservedSuccess as SATISFIED_PRIOR; command and capture goals always retain until they execute.
        """

        agent_state = self.__context.agent_state
        active = agent_state.get_current_sub_goal()
        if active is None or not agent_state.has_sub_goals():
            return ProbeResult(advancement=Advancement(kind=AdvanceKind.RETAIN))

        success = active.success
        verdict = (
            await self.__observe(
                index=active.index, requirement=success.observation, observation=observation
            )
            if isinstance(success, ObservedSuccess)
            else None
        )
        turn = TurnEvidence(
            claim=ClaimEvidence(asserted=False),
            action=ActionEvidence(dispatched=False, executed=False),
            phase=ObservationPhase.PRE_DISPATCH,
            execution=None,
            observation=self.__observed_requirement(success=success),
            verdict=verdict,
        )
        advancement = self.__policy.decide(success=success, evidence=turn)
        if advancement.kind is not AdvanceKind.SATISFIED_PRIOR:
            return ProbeResult(advancement=advancement)

        logger.info(
            "Sub-goal satisfied before dispatch; advancing via SATISFIED_PRIOR",
            extra={
                **self.__log_context(),
                "sub_goal.index": active.index,
                "event": CompletionEvent.CRITERION_OBSERVED.value,
            },
        )
        return ProbeResult(
            advancement=advancement,
            transition=self.__advance_or_complete(current=active, accumulated=[]),
        )

    async def __post_dispatch_evidence(
        self,
        *,
        active: GoalState,
        analysis: AnalysisResult,
        step_result: StepResult,
        binding: Optional[Binding],
        reading: Optional[EffectReading],
        observation: Optional[ScreenObservation],
    ) -> TurnEvidence:
        """
        Build the correlated post-dispatch evidence for the active goal's success variant.

        The observer is consulted only for an ObservedSuccess observation or a CommandSuccess
        postcondition; command-without-postcondition and capture goals rely on the StepResult.
        """

        success = active.success
        requirement = self.__observed_requirement(success=success)
        verdict = (
            await self.__observe(
                index=active.index, requirement=requirement, observation=observation
            )
            if requirement is not None
            else None
        )

        agent_state = self.__context.agent_state
        return TurnEvidence(
            claim=ClaimEvidence(
                asserted=analysis.is_sub_goal_complete or analysis.is_goal_complete
            ),
            action=ActionEvidence(dispatched=True, executed=step_result.executed),
            phase=ObservationPhase.POST_DISPATCH,
            execution=step_result,
            observation=requirement,
            binding=binding,
            effect=reading,
            verdict=verdict,
            validation=analysis.validation,
            stall=self.__stall.assess(effects=agent_state.get_recent_effects()),
        )

    @staticmethod
    def __observed_requirement(*, success: Success) -> Optional[ObservationRequirement]:
        """
        Return the exact observation identity to adjudicate for this success, if any.
        """

        return Eligibility.observation(success=success)

    async def __observe(
        self,
        *,
        index: int,
        requirement: Optional[ObservationRequirement],
        observation: Optional[ScreenObservation],
    ) -> Optional[Verdict]:
        """
        Observe exactly the given requirement on the settled screen and project it to a verdict.
        """

        if requirement is None or observation is None:
            return None

        decision = await self.__criterion_observer.check(
            index=index,
            requirement=requirement,
            observation=observation,
            workflow_id=self.__context.workflow_id,
        )
        return self.__verdict_from(decision=decision)

    @staticmethod
    def __verdict_from(*, decision: Optional[CriterionDecision]) -> Optional[Verdict]:
        """
        Project a fresh criterion decision onto the policy's typed verdict.
        """

        if decision is None:
            return None

        return Verdict(
            outcome=decision.verdict,
            confidence=decision.confidence,
            evidence="; ".join(decision.evidence),
        )

    def __escalate(
        self,
        *,
        active: GoalState,
        advancement: Advancement,
        accumulated: List[StepResult],
    ) -> IntentGraphState:
        """
        Terminate the run when momentum stalls (StallPolicy) instead of looping.
        """

        agent_state = self.__context.agent_state
        reason = (
            CompletionReason.UNSATISFIABLE
            if advancement.kind is AdvanceKind.UNSATISFIABLE
            else CompletionReason.STUCK
        )

        agent_state.mark_complete(reason=reason.value)

        logger.error(
            "Momentum stalled; terminating run",
            extra={
                **self.__log_context(),
                "sub_goal.index": active.index,
                "advancement.kind": advancement.kind.value,
                "completion.reason": reason.value,
                "sub_goal.objective": active.objective[:80],
                "event": "subgoal.escalated",
            },
        )
        return cast(
            "IntentGraphState",
            {
                CommonStateKey.IS_COMPLETE: True,
                IntentStateKey.VERIFY_MODE: None,
                IntentStateKey.SHOULD_RETRY: False,
                IntentStateKey.STEP_RESULTS: accumulated,
                CommonStateKey.COMPLETION_REASON: reason.value,
            },
        )

    def __advance_or_complete(
        self,
        *,
        current: GoalState,
        accumulated: List[StepResult],
    ) -> IntentGraphState:
        """
        Mark non-final sub-goals complete; defer the final commit to VERIFY.
        """

        agent_state = self.__context.agent_state

        if agent_state.has_active_final_sub_goal():
            return self.__route_final_sub_goal_to_verify(current=current, accumulated=accumulated)

        has_more = agent_state.advance_current_sub_goal()

        if has_more:
            return self.__retry_for_next_sub_goal(current=current, accumulated=accumulated)

        raise InvariantViolation(
            "Sub-goal cursor drift: non-final cursor reported no remaining sub-goals."
        )

    def __retry_for_next_sub_goal(
        self,
        *,
        current: GoalState,
        accumulated: List[StepResult],
    ) -> IntentGraphState:
        """
        Emit a graph patch that loops back to GROUND for the next sub-goal.
        """

        agent_state = self.__context.agent_state

        agent_state.clear_verification_loop()
        agent_state.reset_complete_deferrals()
        next_sub_goal = agent_state.get_current_sub_goal()

        logger.info(
            "Sub-goal advanced; looping back to GROUND for next sub-goal",
            extra={
                **self.__log_context(),
                "sub_goal.index": current.index,
                "sub_goal.objective": current.objective[:80],
                "event": CompletionEvent.SUBGOAL_ADVANCED.value,
                "next.sub_goal.index": next_sub_goal.index if next_sub_goal else None,
                "next.sub_goal.objective": (
                    next_sub_goal.objective[:80] if next_sub_goal else None
                ),
            },
        )
        return cast(
            "IntentGraphState",
            {
                IntentStateKey.VERIFY_MODE: None,
                IntentStateKey.SHOULD_RETRY: True,
                IntentStateKey.STEP_RESULTS: accumulated,
            },
        )

    def __route_final_sub_goal_to_verify(
        self,
        *,
        current: GoalState,
        accumulated: List[StepResult],
    ) -> IntentGraphState:
        """
        Route to VERIFY while keeping the final sub-goal active until acceptance.
        """

        completion_reason = "All sub-goals advanced; pending final adjudication"
        self.__context.agent_state.clear_verification_loop()
        self.__context.agent_state.reset_complete_deferrals()

        logger.info(
            "Final sub-goal satisfied by gate; routing to VERIFY without commit",
            extra={
                **self.__log_context(),
                "sub_goal.index": current.index,
                "event": CompletionEvent.INTENT_PENDING.value,
                "sub_goal.objective": current.objective[:80],
                "verify.mode": VerifyMode.PENDING_FINAL_COMMIT.value,
            },
        )
        return cast(
            "IntentGraphState",
            {
                CommonStateKey.IS_COMPLETE: True,
                IntentStateKey.SHOULD_RETRY: False,
                IntentStateKey.STEP_RESULTS: accumulated,
                CommonStateKey.COMPLETION_REASON: completion_reason,
                IntentStateKey.VERIFY_MODE: VerifyMode.PENDING_FINAL_COMMIT.value,
            },
        )

    @staticmethod
    def __analysis_from(*, plan: Any) -> Optional[AnalysisResult]:
        """
        Return the :class:`AnalysisResult` the plan carries in its typed context.
        """

        if not isinstance(plan, PlanResult):
            return None

        return plan.context.analysis

    def __record_executed(
        self,
        *,
        plan: Any,
        active: GoalState,
        receipt: StepResult,
        observation: Optional[ScreenObservation],
        live: Advancement,
    ) -> None:
        """
        Emit the finalized record for a successful dispatch, carrying the real receipt and post-dispatch decision.
        """

        draft = self.__draft(plan=plan)
        if draft is None:
            return
        self.__emit(
            record=self.__runner.finalize_executed(
                draft=draft,
                active=active,
                receipt=receipt,
                live=live,
                screen=self.__post_screen(observation=observation),
                foreground=self.__post_foreground(observation=observation),
                cursor_after=self.__cursor(active=active),
            )
        )

    def __record_failed(
        self,
        *,
        plan: Any,
        active: GoalState,
        receipt: StepResult,
        observation: Optional[ScreenObservation],
    ) -> None:
        """
        Emit the finalized record for a failed dispatch, whose short-circuited live decision is never comparable.
        """

        draft = self.__draft(plan=plan)
        if draft is None:
            return
        self.__emit(
            record=self.__runner.finalize_failed(
                draft=draft,
                active=active,
                receipt=receipt,
                live=Advancement(kind=AdvanceKind.RETAIN, reason=RetainReason.STEP_EXECUTION_FAILED),
                screen=self.__post_screen(observation=observation),
                foreground=self.__post_foreground(observation=observation),
                cursor_after=self.__cursor(active=active),
            )
        )

    @staticmethod
    def __draft(*, plan: Any) -> Optional[ShadowTurnDraft]:
        """
        Return the pre-dispatch draft Analyze attached to the plan, when one is present.
        """

        if not isinstance(plan, PlanResult) or not isinstance(plan.context.shadow, ShadowTurnDraft):
            return None
        return plan.context.shadow

    def __cursor(self, *, active: GoalState) -> GoalCursor:
        """
        Read the active cursor from state after the live decision applied.
        """

        progress = self.__context.agent_state.get_sub_goal_progress()
        if progress is None:
            return GoalCursor(index=active.index, total=active.index + 1)
        index, total = progress
        return GoalCursor(index=index, total=total)

    @staticmethod
    def __post_screen(*, observation: Optional[ScreenObservation]) -> Optional[str]:
        """
        Return the post-dispatch settled-screen identity, never the pre-dispatch screen.
        """

        return observation.hashes.visual_hash if observation is not None else None

    @staticmethod
    def __post_foreground(*, observation: Optional[ScreenObservation]) -> Optional[str]:
        """
        Return the post-dispatch foreground package, when a post observation exists.
        """

        return observation.activity if observation is not None else None

    def __emit(self, *, record: ShadowTurn) -> None:
        """
        Emit a finalized shadow turn through the debug boundary.
        """

        logger.info(
            "Shadow advancement comparison",
            extra={
                **self.__log_context(),
                "event": "shadow.turn.comparison",
                "shadow.record": record.model_dump(mode="json"),
            },
        )

    def __log_adjudicated(
        self,
        *,
        active: GoalState,
        advancement: Advancement,
        step_result: StepResult,
    ) -> None:
        """
        Structured log: advancement decision for this turn.
        """

        logger.info(
            "Advancement adjudicated",
            extra={
                **self.__log_context(),
                "sub_goal.index": active.index,
                "success.kind": active.success.kind.value,
                "advancement.kind": advancement.kind.value,
                "advancement.redispatch": advancement.redispatch,
                "event": CompletionEvent.GATE_ADJUDICATED.value,
                "step.screen_changed": step_result.screen_changed,
                "advancement.reason": (
                    advancement.reason.value if advancement.reason is not None else None
                ),
            },
        )

    def __log_retained(
        self,
        *,
        active: GoalState,
        advancement: Advancement,
        step_result: StepResult,
    ) -> None:
        """
        Structured log: sub-goal retained for another planner turn.
        """

        logger.info(
            "Sub-goal retained; replanning required",
            extra={
                **self.__log_context(),
                "sub_goal.index": active.index,
                "sub_goal.objective": active.objective[:80],
                "event": CompletionEvent.SUBGOAL_RETAINED.value,
                "step.screen_changed": step_result.screen_changed,
                "planner.emitted_action_type": step_result.step.action.action_type.value,
                "advancement.reason": (
                    advancement.reason.value if advancement.reason is not None else None
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
