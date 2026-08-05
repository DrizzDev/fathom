from __future__ import annotations

from typing import Tuple

from fathom.constants.planner import GuardOutcome
from fathom.constants.retries import RetryBranch, RetryKind
from fathom.constants.state import CompletionReason
from fathom.core.agent.state import AgentState
from fathom.core.prompts.rejection import RepeatedFailureRejectionPromptBuilder
from fathom.core.services.vision import VisionService
from fathom.schemas.actions import Action
from fathom.schemas.planner import GuardDecision
from fathom.schemas.results import (
    AnalysisResult,
    PlanContext,
    PlannerRetry,
    PlanResult,
    ToolErrorFeedback,
)
from fathom.schemas.supervision import BlockReason
from fathom.schemas.tools import StateUpdate, ToolArtifact, ToolData, ToolDiagnostic


class RetryFeedback:
    """
    Constructs the single typed retry PlanResult for a turn that must replan instead of executing.
    """

    @staticmethod
    def result(
        *,
        retry: PlannerRetry,
        reason: str,
        analysis: AnalysisResult,
        updates: Tuple[StateUpdate, ...] = (),
        data: Tuple[ToolData, ...] = (),
        artifacts: Tuple[ToolArtifact, ...] = (),
        diagnostics: Tuple[ToolDiagnostic, ...] = (),
    ) -> PlanResult:
        """
        Return a retry PlanResult carrying the typed retry directive and any routed tool response.
        """

        return PlanResult(
            step=None,
            is_complete=False,
            should_retry=True,
            metrics=analysis.metrics,
            memories=analysis.memories,
            reason=reason,
            context=PlanContext(retry=retry),
            updates=updates,
            data=data,
            artifacts=artifacts,
            diagnostics=diagnostics,
        )

    @classmethod
    def no_action(
        cls,
        *,
        analysis: AnalysisResult,
        reason: str,
        updates: Tuple[StateUpdate, ...] = (),
        data: Tuple[ToolData, ...] = (),
        artifacts: Tuple[ToolArtifact, ...] = (),
        diagnostics: Tuple[ToolDiagnostic, ...] = (),
    ) -> PlanResult:
        """
        Return the retry result for a turn the model left without an executable action.
        """

        return cls.result(
            retry=PlannerRetry(kind=RetryKind.SILENT_REJECTION, branch=RetryBranch.UNKNOWN),
            reason=reason,
            analysis=analysis,
            updates=updates,
            data=data,
            artifacts=artifacts,
            diagnostics=diagnostics,
        )


class RejectionFeedback:
    """
    Turns a rejected guard decision or command validation into model feedback and the typed retry result.
    """

    def __init__(self, *, vision: VisionService) -> None:
        """
        Bind the vision service used to author rejection history for the model's next turn.
        """

        self.__vision = vision

    def reject(
        self,
        *,
        decision: GuardDecision,
        state: AgentState,
        action: Action,
        analysis: AnalysisResult,
        interactive_mode: bool,
    ) -> PlanResult:
        """
        Author rejection history and the typed retry result for a blocking guard decision.
        """

        descriptor = action.to_description()

        if decision.outcome is GuardOutcome.SILENT_REJECTION:
            self.__reject_with(
                state=state,
                analysis=analysis,
                prompt=RepeatedFailureRejectionPromptBuilder.build(
                    interactive=interactive_mode, action_descriptor=descriptor
                ),
            )
            return RetryFeedback.result(
                retry=PlannerRetry(
                    kind=RetryKind.SILENT_REJECTION,
                    branch=RetryBranch.SHOULD_AVOID_ACTION,
                    action=descriptor,
                ),
                reason=CompletionReason.FAILED.value,
                analysis=analysis,
            )

        if decision.outcome is GuardOutcome.CURRENT_SCREEN_REPEAT:
            reason = decision.reason or ""
            state.record_blocked_action(
                action=action,
                reason=reason,
                block_reason=BlockReason.REPEATED_CURRENT_SCREEN_ACTION,
            )
            self.__reject_with(
                state=state,
                analysis=analysis,
                prompt=(
                    f"REJECTED: {reason} Choose a different action "
                    "that advances the active sub-goal on the current screen, or ask "
                    "the user if the screen contradicts the sub-goal."
                ),
            )
            return RetryFeedback.result(
                retry=PlannerRetry(
                    kind=RetryKind.LLM_FEEDBACK,
                    branch=RetryBranch.CURRENT_SCREEN_REPEAT,
                    action=descriptor,
                    block=BlockReason.REPEATED_CURRENT_SCREEN_ACTION,
                ),
                reason=CompletionReason.ACTION_BLOCKED.value,
                analysis=analysis,
            )

        state.record_repeated_action_failure(action=action)
        self.__reject_with(
            state=state,
            analysis=analysis,
            prompt=(
                f"REJECTED: '{descriptor}' was repeated 3+ times on the same screen "
                "without progress. You MUST choose a completely different action."
            ),
        )
        return RetryFeedback.result(
            retry=PlannerRetry(
                kind=RetryKind.LLM_FEEDBACK,
                branch=RetryBranch.IS_ACTION_REPEATING_ON_SCREEN,
                action=descriptor,
            ),
            reason=CompletionReason.ACTION_BLOCKED.value,
            analysis=analysis,
        )

    def reject_command(
        self, *, state: AgentState, analysis: AnalysisResult, feedback: ToolErrorFeedback
    ) -> PlanResult:
        """
        Author rejection history and the typed retry result for a command that failed structural validation.
        """

        self.__reject_with(state=state, analysis=analysis, prompt=f"REJECTED: {feedback.message}")
        return RetryFeedback.result(
            retry=PlannerRetry(kind=RetryKind.SILENT_REJECTION, branch=RetryBranch.UNKNOWN),
            reason=feedback.message,
            analysis=analysis,
        )

    def __reject_with(self, *, state: AgentState, analysis: AnalysisResult, prompt: str) -> None:
        """
        Record the rejection history the vision turn replays to the model on the next turn.
        """

        state.set_rejection_history(
            self.__vision.build_rejection_history_from_analysis(
                analysis=analysis, rejection_reason=prompt
            )
        )
