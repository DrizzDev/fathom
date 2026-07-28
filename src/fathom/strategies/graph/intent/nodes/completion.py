from __future__ import annotations

from logging import getLogger
from typing import Any, Dict, List, Optional, cast

from fathom.constants.capability import CompletionMode
from fathom.constants.completion import GateOutcome
from fathom.constants.observability import CompletionEvent
from fathom.constants.state import (
    CommonStateKey,
    CompletionReason,
    IntentStateKey,
    PlanMetadataKey,
    VerifyMode,
)
from fathom.constants.turn.advancement import AdvanceKind
from fathom.core.agent.advancement import AdvancementPolicy
from fathom.core.agent.capture import StoreCaptureCompletionPolicy
from fathom.core.agent.stall import StallPolicy
from fathom.core.exceptions import InvariantViolation
from fathom.core.services.criterion import CriterionObserver
from fathom.core.services.directive import DirectivePolicy
from fathom.schemas.advancement import Advancement, RetainHistory
from fathom.schemas.binding import Binding
from fathom.schemas.completion import (
    ActionEvidence,
    ClaimEvidence,
    CompletionEvidence,
    GateDecision,
    ScreenEvidence,
    ValidationEvidence,
)
from fathom.schemas.criterion import CriterionDecision
from fathom.schemas.effect import ActionEffect, ActionEffectStatus, EffectReading
from fathom.schemas.observability import CompletionLogContext
from fathom.schemas.observation import ScreenObservation
from fathom.schemas.reasoning import SubGoalCompletionSignal
from fathom.schemas.results import AnalysisResult, PlanResult
from fathom.schemas.steps import StepResult
from fathom.schemas.subgoal import SubGoal
from fathom.schemas.turn import TurnEvidence
from fathom.schemas.vision import ActionKind, ActionKindResolver
from fathom.strategies.graph.context import GraphContext
from fathom.strategies.graph.state import IntentGraphState

logger = getLogger(__name__)


class SubGoalEvaluator:
    """
    Decide whether the executed step satisfies the active sub-goal.

    Each turn assembles the typed TurnEvidence for the sub-goal (claim,
    action, binding, effect, validation, verdict, stall) and the
    AdvancementPolicy adjudicates it against the sub-goal's projected
    completion mode:

      - Observed satisfaction (the vision verdict, or a canonical
        validation) advances; a model claim only advances when the effect
        corroborates it, and never against an observed refutation.
      - Consecutive retention is bounded by the policy backstop: once the
        retain streak is exhausted the turn escalates, terminating the run
        instead of looping.

    Capture sub-goals are decided upstream by the StoreCaptureCompletionPolicy
    and never reach the advancement policy. The CriterionObserver stays as an
    additive RCA-grade signal, logged on every decision but never gating.
    """

    def __init__(
        self,
        *,
        context: GraphContext,
        criterion_observer: CriterionObserver,
        policy: Optional[AdvancementPolicy] = None,
        capture_policy: Optional[StoreCaptureCompletionPolicy] = None,
    ) -> None:
        """
        Bind the evaluator to its graph context, criterion observer, advancement policy, and capture policy.
        """

        self.__context = context
        self.__criterion_observer = criterion_observer
        self.__policy = policy if policy is not None else AdvancementPolicy()
        self.__capture_policy = (
            capture_policy if capture_policy is not None else StoreCaptureCompletionPolicy()
        )

        self.__stall = StallPolicy()
        self.__projection: Optional[DirectivePolicy] = None

    def __projector(self) -> DirectivePolicy:
        """
        Build the task projection lazily against the live catalog.
        """

        if self.__projection is None:
            self.__projection = DirectivePolicy(catalog=self.__context.catalog)

        return self.__projection

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

        if not self.__is_evaluable(current=current, has_sub_goals=agent_state.has_sub_goals()):
            return None

        if not step_result.success:
            self.__log_skipped(reason="step.failed", step_result=step_result)
            return None

        analysis = self.__analysis_from(plan=plan)
        if analysis is None:
            return None

        active = cast("SubGoal", current)
        emitted_kind = ActionKindResolver.resolve(action_type=step_result.step.action.action_type)

        if self.__requires_capture_completion(active=active):
            gate_decision = self.__capture_policy.evaluate(
                step_result=step_result,
                capture_store=self.__context.capture_store,
            )
            evidence = self.__capture_evidence(step_result=step_result, decision=gate_decision)
            advancement = self.__from_capture(decision=gate_decision)
        else:
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
                execution_success=step_result.executed,
                criterion_decision=criterion_decision,
                screen_changed=step_result.screen_changed,
                screen_description=step_result.observation or step_result.step.action.target or "",
            )
            self.__log_evidence_assessed(
                active=active,
                evidence=evidence,
                effect=last_effect,
                step_result=step_result,
            )

            turn = TurnEvidence(
                effect=reading,
                binding=binding,
                claim=evidence.claim,
                action=evidence.action,
                validation=analysis.validation,
                stall=self.__stall.assess(effects=agent_state.get_recent_effects()),
            )
            advancement = self.__policy.decide(
                task=self.__projector().project(sub_goal=active),
                evidence=turn,
                history=RetainHistory(consecutive=agent_state.subgoal_retain_streak),
            )
            self.__reconcile_streak(advancement=advancement)
            self.__log_adjudicated(
                active=active,
                evidence=evidence,
                advancement=advancement,
                effect=last_effect,
                step_result=step_result,
            )

        if advancement.kind in (AdvanceKind.ADVANCE, AdvanceKind.SATISFIED_PRIOR):
            signal = self.__build_storage_signal(
                active=active,
                analysis=analysis,
                step_result=step_result,
            )
            return self.__advance_or_complete(
                signal=signal,
                current=active,
                kind=emitted_kind,
                evidence=evidence,
                accumulated=accumulated,
            )

        if advancement.kind in (AdvanceKind.ESCALATE, AdvanceKind.UNSATISFIABLE):
            return self.__escalate(
                active=active,
                advancement=advancement,
                accumulated=accumulated,
            )

        self.__log_retained(
            active=active,
            evidence=evidence,
            advancement=advancement,
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
            screen_description=step_result.observation or step_result.step.action.target or "",
        )

    @staticmethod
    def __is_evaluable(*, current: Optional[SubGoal], has_sub_goals: bool) -> bool:
        """
        Whether a sub-goal context exists for evaluation this turn.
        """

        return current is not None and has_sub_goals

    def __requires_capture_completion(self, *, active: SubGoal) -> bool:
        """
        Return whether the active directive can only advance through captured evidence.
        """

        if active.directive is None:
            return False

        directed = self.__context.catalog.profile(action_type=active.directive).completion
        return directed is CompletionMode.CAPTURE_VERIFIED

    @staticmethod
    def __capture_evidence(
        *, step_result: StepResult, decision: GateDecision
    ) -> CompletionEvidence:
        """
        Build observability-only evidence for a capture turn; the decision itself comes from the policy.
        """

        request = step_result.step.action.capture

        if decision.outcome is GateOutcome.ADVANCE and request is not None:
            note = f"capture.verified: stored '{request.name}'"

        elif decision.retain_reason is not None:
            note = f"capture.retained: {decision.retain_reason.value}"

        else:
            note = "capture.retained"

        return CompletionEvidence(
            notes=(note,),
            screen=ScreenEvidence(evolved=False),
            claim=ClaimEvidence(asserted=False),
            action=ActionEvidence(dispatched=False, executed=step_result.executed),
            validation=ValidationEvidence(executed=False),
        )

    @staticmethod
    def __from_capture(*, decision: GateDecision) -> Advancement:
        """
        Project a capture-policy gate decision onto the shared advancement vocabulary.
        """

        if decision.outcome is GateOutcome.ADVANCE:
            return Advancement(kind=AdvanceKind.ADVANCE)

        return Advancement(kind=AdvanceKind.RETAIN, reason=decision.retain_reason)

    def __reconcile_streak(self, *, advancement: Advancement) -> None:
        """
        Persist the active sub-goal's retain streak so the backstop survives a checkpoint resume.
        """

        if advancement.kind is AdvanceKind.RETAIN:
            self.__context.agent_state.record_subgoal_retain()
            return

        self.__context.agent_state.reset_subgoal_retain()

    def __escalate(
        self,
        *,
        active: SubGoal,
        advancement: Advancement,
        accumulated: List[StepResult],
    ) -> IntentGraphState:
        """
        Terminate the run when the retain backstop is exhausted instead of looping.
        """

        agent_state = self.__context.agent_state
        reason = (
            CompletionReason.UNSATISFIABLE
            if advancement.kind is AdvanceKind.UNSATISFIABLE
            else CompletionReason.STUCK
        )

        agent_state.reset_subgoal_retain()
        agent_state.mark_complete(reason=reason.value)

        logger.error(
            "Sub-goal retain backstop exhausted; terminating run",
            extra={
                **self.__log_context(),
                "sub_goal.index": active.index,
                "sub_goal.kind": active.kind.value,
                "advancement.kind": advancement.kind.value,
                "completion.reason": reason.value,
                "sub_goal.description": active.description[:80],
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
        current: SubGoal,
        kind: ActionKind,
        evidence: CompletionEvidence,
        accumulated: List[StepResult],
        signal: SubGoalCompletionSignal,
    ) -> IntentGraphState:
        """
        Mark non-final sub-goals complete; defer final commit to VERIFY.
        """

        agent_state = self.__context.agent_state

        if agent_state.has_active_final_sub_goal():
            return self.__route_final_sub_goal_to_verify(
                kind=kind,
                current=current,
                evidence=evidence,
                accumulated=accumulated,
            )

        has_more = agent_state.mark_current_sub_goal_complete(completion_signal=signal)

        if has_more:
            return self.__retry_for_next_sub_goal(
                current=current, evidence=evidence, accumulated=accumulated, kind=kind
            )

        raise InvariantViolation(
            "Sub-goal cursor drift: non-final cursor reported no remaining sub-goals."
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

        agent_state.clear_verification_loop()
        agent_state.reset_complete_deferrals()
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
                IntentStateKey.VERIFY_MODE: None,
                IntentStateKey.SHOULD_RETRY: True,
                IntentStateKey.STEP_RESULTS: accumulated,
            },
        )

    def __route_final_sub_goal_to_verify(
        self,
        *,
        current: SubGoal,
        kind: ActionKind,
        evidence: CompletionEvidence,
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
                "kind": kind.value,
                "sub_goal.index": current.index,
                "sub_goal.kind": current.kind.value,
                "evidence.notes": list(evidence.notes),
                "event": CompletionEvent.INTENT_PENDING.value,
                "sub_goal.description": current.description[:80],
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
                "action.dispatched": evidence.action.dispatched,
                "validation.executed": evidence.validation.executed,
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

    def __log_adjudicated(
        self,
        *,
        active: SubGoal,
        advancement: Advancement,
        step_result: StepResult,
        evidence: CompletionEvidence,
        effect: Optional[ActionEffect],
    ) -> None:
        """
        Structured log: advancement decision for this turn and the typed signals behind it.
        """

        logger.info(
            "Advancement adjudicated",
            extra={
                **self.__log_context(),
                "sub_goal.index": active.index,
                "sub_goal.kind": active.kind.value,
                "advancement.kind": advancement.kind.value,
                "advancement.redispatch": advancement.redispatch,
                "screen.evolved": evidence.screen.evolved,
                "claim.asserted": evidence.claim.asserted,
                "action.dispatched": evidence.action.dispatched,
                "validation.executed": evidence.validation.executed,
                "event": CompletionEvent.GATE_ADJUDICATED.value,
                "step.screen_changed": step_result.screen_changed,
                "advancement.reason": (
                    advancement.reason.value if advancement.reason is not None else None
                ),
                "retain.streak": self.__context.agent_state.subgoal_retain_streak,
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
        advancement: Advancement,
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
                "retain.streak": self.__context.agent_state.subgoal_retain_streak,
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
