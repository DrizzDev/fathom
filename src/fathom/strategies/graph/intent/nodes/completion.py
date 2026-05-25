from __future__ import annotations

from logging import getLogger
from typing import Any, Dict, List, Optional, cast

from fathom.constants.reasoning import (
    SUB_GOAL_COMPLETION_REQUIRED_SIGNALS,
    VALIDATION_KEYWORDS,
)
from fathom.constants.state import CommonStateKey, IntentStateKey, PlanMetadataKey
from fathom.schemas.reasoning import SubGoalCompletionSignal
from fathom.schemas.results import AnalysisResult, PlanResult
from fathom.schemas.steps import StepResult
from fathom.schemas.subgoal import SubGoalKind
from fathom.strategies.graph.context import GraphContext
from fathom.strategies.graph.state import IntentGraphState

logger = getLogger(__name__)


class SubGoalEvaluator:
    """
    Classifies sub-goals and applies the coords-style post-action completion signal gate.
    """

    def __init__(
        self,
        *,
        context: GraphContext,
    ) -> None:
        """
        Initialize with the shared graph context.
        """

        self.__context = context

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

    async def evaluate(
        self,
        *,
        plan: Any,
        step_result: StepResult,
        accumulated: List[StepResult],
    ) -> Optional[IntentGraphState]:
        """
        Emit a graph patch when the executed action clears the completion signal gate.
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

        index, total = agent_state.get_sub_goal_progress()
        if not self.__meets_threshold(signal=signal):
            self.__log_incomplete(
                kind=kind,
                signal=signal,
                current_index=current.index,
                progress=(index + 1, total),
                description=current.description,
            )
            return None

        logger.info(
            "Sub-goal signals passed; routing to VERIFY before advancing",
            extra={
                **self.__log_context(),
                "kind": kind.value,
                "event": "subgoal.verify.pending",
                "signal.count": self.__signal_count(signal=signal),
                "sub_goal.index": current.index,
                "sub_goal.description": current.description[:80],
            },
        )
        agent_state.mark_complete(
            reason=f"Sub-goal '{current.description[:50]}' pending verification"
        )
        return cast(
            "IntentGraphState",
            {
                CommonStateKey.IS_COMPLETE: True,
                CommonStateKey.COMPLETION_REASON: (
                    f"Sub-goal '{current.description[:50]}' pending verification"
                ),
                IntentStateKey.STEP_RESULTS: accumulated,
            },
        )

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

    def __log_incomplete(
        self,
        *,
        signal: SubGoalCompletionSignal,
        description: str,
        kind: SubGoalKind,
        current_index: int,
        progress: tuple[int, int],
    ) -> None:
        """
        Emit a single structured verdict log for incomplete sub-goal checks.
        """

        index, total = progress
        logger.info(
            "Sub-goal completion check did not pass",
            extra={
                **self.__log_context(),
                "kind": kind.value,
                "progress.total": total,
                "event": "subgoal.incomplete",
                "progress.current": index,
                "signal.count": self.__signal_count(signal=signal),
                "required.signals": SUB_GOAL_COMPLETION_REQUIRED_SIGNALS,
                "sub_goal.index": current_index,
                "sub_goal.description": description[:80],
                "signal.claim_verified": signal.claim_verified,
                "signal.action_effective": signal.action_effective,
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

    @classmethod
    def __meets_threshold(cls, *, signal: SubGoalCompletionSignal) -> bool:
        """
        Return whether enough independent sub-goal completion signals agree.
        """

        return cls.__signal_count(signal=signal) >= SUB_GOAL_COMPLETION_REQUIRED_SIGNALS

    @staticmethod
    def __signal_count(*, signal: SubGoalCompletionSignal) -> int:
        """
        Count independent positive completion signals.
        """

        return sum(
            (
                signal.claim_verified,
                signal.action_effective,
                signal.keyword_match,
                signal.trace_verified,
            )
        )
