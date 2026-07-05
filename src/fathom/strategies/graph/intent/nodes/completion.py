from __future__ import annotations

import asyncio
from logging import getLogger
from typing import Any, Dict, List, Optional, cast

from fathom.constants.capability import CompletionMode
from fathom.constants.completion import AdvanceReason, GateOutcome
from fathom.constants.observability import CompletionEvent
from fathom.constants.state import CommonStateKey, IntentStateKey, PlanMetadataKey, VerifyMode
from fathom.core.agent.capture import StoreCaptureCompletionPolicy
from fathom.core.agent.completion import CompletionGate
from fathom.core.exceptions import InvariantViolation
from fathom.core.services.criterion import CriterionObserver
from fathom.schemas.completion import (
    ActionEvidence,
    ClaimEvidence,
    CompletionEvidence,
    GateDecision,
    ScreenEvidence,
)
from fathom.schemas.criterion import CriterionDecision
from fathom.schemas.effect import ActionEffect, ActionEffectStatus
from fathom.schemas.observability import CompletionLogContext
from fathom.schemas.observation import ScreenObservation
from fathom.schemas.reasoning import SubGoalCompletionSignal
from fathom.schemas.results import AnalysisResult, PlanResult
from fathom.schemas.steps import StepResult
from fathom.schemas.subgoal import SubGoal, SubGoalKind
from fathom.schemas.vision import ActionKind, action_kind_for
from fathom.strategies.graph.context import GraphContext
from fathom.strategies.graph.state import IntentGraphState

logger = getLogger(__name__)


class SubGoalEvaluator:
    """
    Decide whether the executed step satisfies the active sub-goal.

    Each turn produces a typed CompletionEvidence bundle (claim, action,
    screen, optional criterion) and the emitted action kind, which the
    CompletionGate adjudicates per sub-goal kind:

      - ACTION sub-goals require asserted claim AND justified rationale AND
        a dispatched action that caused screen evolution. The single
        exception is the VALIDATION-kind escape branch: when the planner
        emits a VALIDATE action against an ACTION sub-goal and asserts
        completion, the gate advances on (asserted AND dispatched) alone.
        VALIDATE is a read action that cannot move the screen; requiring
        screen.evolved for this branch would loop indefinitely whenever
        the world is already past the failed step.
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
        capture_policy: Optional[StoreCaptureCompletionPolicy] = None,
    ) -> None:
        """
        Bind the evaluator to its graph context, criterion observer, gate, and capture policy.
        """

        self.__context = context
        self.__criterion_observer = criterion_observer
        self.__gate = gate if gate is not None else CompletionGate()
        self.__capture_policy = (
            capture_policy if capture_policy is not None else StoreCaptureCompletionPolicy()
        )

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
        emitted_kind = action_kind_for(step_result.step.action.action_type)

        if self.__requires_capture_completion(active=active):
            decision = self.__capture_policy.evaluate(
                step_result=step_result,
                capture_store=self.__context.capture_store,
            )
            evidence = self.__capture_evidence(step_result=step_result, decision=decision)
        else:
            criterion_decision = await self.__observe_criterion(
                active=active,
                observation=observation,
                step_result=step_result,
            )

            last_effect = agent_state.get_last_action_effect()

            directive = agent_state.operator_directive
            directive_kind = (
                directive.kind
                if directive is not None and agent_state.has_active_directive
                else None
            )

            semantic_similarity = await self.__semantic_similarity(
                sub_goal=active,
                analysis=analysis,
            )

            evidence = self.__context.reasoner.assess_completion(
                sub_goal=active,
                analysis=analysis,
                effect=last_effect,
                directive_kind=directive_kind,
                execution_success=step_result.executed,
                semantic_similarity=semantic_similarity,
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

            decision = self.__gate.adjudicate(
                evidence=evidence,
                sub_goal=active,
                action_kind=emitted_kind,
            )
            self.__log_gate_adjudicated(
                active=active,
                evidence=evidence,
                decision=decision,
                effect=last_effect,
                step_result=step_result,
                action_kind=emitted_kind,
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
                kind=emitted_kind,
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
            claim=ClaimEvidence(asserted=False, justified=False),
            action=ActionEvidence(dispatched=False, executed=step_result.executed),
        )

    async def __semantic_similarity(
        self,
        *,
        sub_goal: SubGoal,
        analysis: AnalysisResult,
    ) -> Optional[float]:
        """
        Cosine similarity between rationale and sub-goal via embedding port + cache; ``None`` on any failure.
        """

        embedder = self.__context.embedder
        cache = self.__context.embedding_cache

        if cache is None or embedder is None:
            logger.info(
                "Semantic similarity unavailable; embedder or cache missing",
                extra={
                    "component": "graph.intent.completion",
                    "event": "completion.semantic_similarity.unavailable",
                    "cache.present": cache is not None,
                    "embedder.present": embedder is not None,
                },
            )
            return None

        rationale = (analysis.subgoal_completion_reason or analysis.reasoning or "").strip()
        if not rationale:
            logger.info(
                "Semantic similarity skipped; rationale text is empty",
                extra={
                    "component": "graph.intent.completion",
                    "event": "completion.semantic_similarity.empty_rationale",
                    "sub_goal.index": sub_goal.index,
                },
            )
            return None

        try:
            sub_goal_vector = await cache.get(text=sub_goal.description)
            if sub_goal_vector is None:
                logger.info(
                    "Sub-goal embedding cache miss",
                    extra={
                        "component": "graph.intent.completion",
                        "event": "completion.semantic_similarity.cache_miss",
                        "sub_goal.index": sub_goal.index,
                    },
                )
                return None
            rationale_result = await embedder.embed(texts=(rationale,))
        except asyncio.CancelledError:
            raise
        except Exception as exception:  # noqa: BLE001 - logged with kind + message
            logger.warning(
                "Semantic similarity scoring failed; falling back to legacy verifier",
                extra={
                    "component": "graph.intent.completion",
                    "event": "completion.semantic_similarity.error",
                    "sub_goal.index": sub_goal.index,
                    "error.kind": type(exception).__name__,
                    "error.message": str(exception),
                },
            )
            return None

        if not rationale_result.vectors:
            logger.warning(
                "Rationale embedding returned no vectors",
                extra={
                    "component": "graph.intent.completion",
                    "event": "completion.semantic_similarity.empty_result",
                    "sub_goal.index": sub_goal.index,
                },
            )
            return None

        try:
            score = float(sub_goal_vector.cosine(other=rationale_result.vectors[0]))
        except ValueError as exception:
            logger.warning(
                "Embedding vector dimension mismatch",
                extra={
                    "component": "graph.intent.completion",
                    "event": "completion.semantic_similarity.dimension_mismatch",
                    "sub_goal.index": sub_goal.index,
                    "error.message": str(exception),
                },
            )
            return None

        logger.info(
            "Semantic similarity resolved",
            extra={
                "component": "graph.intent.completion",
                "event": "completion.semantic_similarity.resolved",
                "similarity.score": score,
                "sub_goal.index": sub_goal.index,
                "rationale.length": len(rationale),
            },
        )
        return score

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
        action_kind: ActionKind,
        evidence: CompletionEvidence,
        effect: Optional[ActionEffect],
    ) -> None:
        """
        Structured log: completion-gate decision for this turn, including which branch ratified an ADVANCE.
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
                "gate.advance_reason": self.__advance_reason(
                    sub_goal=active,
                    decision=decision,
                    evidence=evidence,
                    action_kind=action_kind,
                ),
                "action.kind": action_kind.value,
                "effect.status": (effect.status.value if effect is not None else None),
                "veto.applied": self.__no_progress_vetoed(
                    effect=effect,
                    screen_evolved=evidence.screen.evolved,
                    screen_changed=step_result.screen_changed,
                ),
            },
        )

    @staticmethod
    def __advance_reason(
        *,
        sub_goal: SubGoal,
        decision: GateDecision,
        action_kind: ActionKind,
        evidence: CompletionEvidence,
    ) -> Optional[str]:
        """
        Tag an ADVANCE outcome with the branch that ratified it so RCA can distinguish strict vs implicit-completion.
        """

        if decision.outcome is not GateOutcome.ADVANCE:
            return None

        if (
            not evidence.screen.evolved
            and sub_goal.kind is SubGoalKind.ACTION
            and action_kind is ActionKind.VALIDATION
        ):
            return AdvanceReason.VALIDATION_IMPLICIT_COMPLETION.value

        return AdvanceReason.STRICT_PATH.value

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
