from __future__ import annotations

from logging import getLogger
from typing import Any, Dict, List, Optional, Tuple, cast

from fathom.constants.reasoning import VALIDATION_KEYWORDS
from fathom.constants.screen import NO_PROGRESS_RECOVERY_THRESHOLD
from fathom.constants.state import CommonStateKey, IntentStateKey, PlanMetadataKey
from fathom.core.recovery import RecoveryTrigger
from fathom.core.runtime.adapter import ExecutionTaskAdapter
from fathom.schemas.completion import CompletionVerdict
from fathom.schemas.events import RuntimeEventKind
from fathom.schemas.outcomes import ActionOutcome, OutcomeStatus
from fathom.schemas.results import AnalysisResult, PlanResult
from fathom.schemas.screens import ScreenCapture
from fathom.schemas.steps import StepResult
from fathom.schemas.subgoal import SubGoalKind
from fathom.schemas.tasks import TaskStatus
from fathom.strategies.graph.context import GraphContext
from fathom.strategies.graph.intent.nodes.recovery import RecoveryDispatcher
from fathom.strategies.graph.state import IntentGraphState

logger = getLogger(__name__)


class SubGoalEvaluator:
    """
    Classifies sub-goals, applies the completion floor, and routes stuck recoveries.
    """

    def __init__(
        self,
        *,
        context: GraphContext,
        recovery: RecoveryDispatcher,
    ) -> None:
        """
        Initialize with the shared graph context and the recovery dispatcher.
        """

        self.__context = context
        self.__recovery = recovery

    @staticmethod
    def classify(
        *,
        description: str,
        step_result: StepResult,
    ) -> SubGoalKind:
        """
        Classify a sub-goal as a validation step or an action step.
        """

        if step_result.step.event_type == "validation":
            return SubGoalKind.VALIDATION

        if any(keyword in description.lower() for keyword in VALIDATION_KEYWORDS):
            return SubGoalKind.VALIDATION

        return SubGoalKind.ACTION

    @staticmethod
    def floor(
        *,
        kind: SubGoalKind,
        analysis: AnalysisResult,
        outcome: Optional[ActionOutcome],
    ) -> Optional[str]:
        """
        Return a reason string when the completion floor blocks advancement.
        """

        if kind == SubGoalKind.VALIDATION:
            return None

        task_status = analysis.task_status

        if task_status == TaskStatus.BLOCKED:
            return "Model reported task BLOCKED; supervision and healing decide the next step."

        if task_status == TaskStatus.NOT_MET:
            return "Model reported task NOT_MET; refusing to advance on legacy gate alone."

        if task_status == TaskStatus.MET:
            if outcome is None or outcome.status != OutcomeStatus.EFFECTIVE:
                return "Model reported task MET but observed outcome is not effective."
            return None

        if outcome is None or outcome.status == OutcomeStatus.NO_EFFECT:
            return "Observed outcome reports no effect; refusing to advance on self-report alone."

        return None

    async def evaluate(
        self,
        *,
        plan: Any,
        step_result: StepResult,
        accumulated: List[StepResult],
        outcome: Optional[ActionOutcome] = None,
    ) -> Optional[IntentGraphState]:
        """
        Apply the completion floor and emit a graph patch when a sub-goal advances.
        """

        agent_state = self.__context.agent_state
        current = agent_state.get_current_sub_goal()

        if current is None or not agent_state.has_sub_goals():
            return None

        if not step_result.success:
            logger.info(
                "Skipping sub-goal completion check on failed step",
                extra={
                    **self.__log_context(),
                    "reason": "step.failed",
                    "error.message": step_result.error,
                    "event": "subgoal.evaluate.skipped",
                },
            )
            return None

        analysis = self.__analysis_from(plan=plan)
        if analysis is None:
            return None

        kind = self.classify(step_result=step_result, description=current.description)

        signal = self.__context.reasoner.analyze_subgoal_completion(
            analysis=analysis,
            sub_goal_description=current.description,
            delta_score=agent_state.last_delta_score,
            screen_changed=step_result.screen_changed or kind == SubGoalKind.VALIDATION,
            screen_description=step_result.observation or step_result.step.action.target or "",
        )

        task = ExecutionTaskAdapter().from_sub_goal(sub_goal=current, kind=kind)

        verdict = self.__context.completion_service.evaluate(
            task=task,
            outcome=outcome,
            status=TaskStatus.MET if signal.claim_verified else None,
        )
        if (reason := self.floor(kind=kind, analysis=analysis, outcome=outcome)) is not None:
            verdict = verdict.model_copy(update={"complete": False, "reason": reason})

        index, total = agent_state.get_sub_goal_progress()

        self.__log_verdict(
            kind=kind,
            signal=signal,
            verdict=verdict,
            current_index=current.index,
            progress=(index + 1, total),
            description=current.description,
        )

        if not verdict.complete:
            return None

        has_more = agent_state.mark_current_sub_goal_complete(completion_signal=signal)
        await self.__context.event_emitter.emit(
            kind=RuntimeEventKind.TASK_UPDATED,
            step=agent_state.step_count,
            payload={
                "task.has_more": has_more,
                "task.kind": task.kind.value,
                "task.complete": verdict.complete,
                "task.identifier": task.identifier,
                "task.criterion": task.criterion[:120],
                "task.next_state": verdict.next_state.value,
            },
        )

        if has_more:
            return cast(
                "IntentGraphState",
                {
                    IntentStateKey.SHOULD_RETRY: True,
                    IntentStateKey.STEP_RESULTS: accumulated,
                },
            )

        agent_state.mark_complete(reason="All sub-goals completed sequentially")
        return cast(
            "IntentGraphState",
            {
                CommonStateKey.IS_COMPLETE: True,
                IntentStateKey.STEP_RESULTS: accumulated,
                CommonStateKey.COMPLETION_REASON: "All sub-goals completed sequentially",
            },
        )

    async def recover_if_stuck(
        self,
        *,
        capture: ScreenCapture,
        step_result: StepResult,
    ) -> Optional[IntentGraphState]:
        """
        Dispatch recovery when the agent is stuck, making no progress, or over budget.
        """

        agent_state = self.__context.agent_state
        hint = step_result.step.action.target

        if agent_state.is_stuck:
            return await self.__recovery.try_recover(
                hint=hint,
                capture=capture,
                trigger=RecoveryTrigger.LOOP_DETECTED,
                reason="loop detector observed repetition without progress",
            )

        if agent_state.consecutive_no_progress_count >= NO_PROGRESS_RECOVERY_THRESHOLD:
            return await self.__recovery.try_recover(
                hint=hint,
                capture=capture,
                trigger=RecoveryTrigger.NO_PROGRESS,
                reason=(
                    f"{agent_state.consecutive_no_progress_count} consecutive actions "
                    "produced no measurable visual progress"
                ),
            )

        if agent_state.current_sub_goal_over_budget:
            return await self.__recovery.try_recover(
                hint=hint,
                capture=capture,
                trigger=RecoveryTrigger.SUBGOAL_BUDGET_EXCEEDED,
                reason=(
                    f"sub-goal exhausted step budget "
                    f"({agent_state.current_sub_goal_action_count} actions) "
                    "without its success criterion being met"
                ),
            )

        return None

    @staticmethod
    def __analysis_from(*, plan: Any) -> Optional[AnalysisResult]:
        """
        Reconstruct the AnalysisResult attached to plan metadata.
        """

        if not isinstance(plan, PlanResult) or not plan.metadata:
            return None

        raw = plan.metadata.get(PlanMetadataKey.ANALYSIS.value)
        if raw is None:
            return None

        return raw if isinstance(raw, AnalysisResult) else AnalysisResult.model_validate(raw)

    def __log_verdict(
        self,
        *,
        signal: Any,
        description: str,
        kind: SubGoalKind,
        current_index: int,
        progress: Tuple[int, int],
        verdict: CompletionVerdict,
    ) -> None:
        """
        Emit a single structured verdict log for both success and skip paths.
        """

        index, total = progress
        logger.info(
            "Sub-goal verdict computed",
            extra={
                **self.__log_context(),
                "kind": kind.value,
                "progress.total": total,
                "reason": verdict.reason,
                "event": "subgoal.verdict",
                "progress.current": index,
                "complete": verdict.complete,
                "sub_goal.index": current_index,
                "sub_goal.description": description[:80],
                "missing": [item.value for item in verdict.missing],
                "signal.claim_verified": getattr(signal, "claim_verified", None),
                "signal.action_effective": getattr(signal, "action_effective", None),
            },
        )

    def __log_context(self) -> Dict[str, Any]:
        """
        Return shared structured-logging context for sub-goal entries.
        """

        return {
            "component": "graph.intent.completion",
            "workflow.id": self.__context.workflow_id,
        }
