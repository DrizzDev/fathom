from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class SubGoalCompletionSignal(BaseModel):
    """
    Multi-signal verification for sub-goal completion.
    Mirrors CompletionSignal but scoped to individual sub-goals.
    """

    evidence: str = Field(default="", description="Textual evidence of sub-goal completion")
    llm_confidence: float = Field(default=0.0, description="Confidence score from LLM")

    keyword_match: bool = Field(
        default=False, description="Whether sub-goal keywords matched reasoning"
    )
    action_executed: bool = Field(
        default=False, description="Whether action was executed for this sub-goal"
    )
    llm_signaled: bool = Field(
        default=False,
        description="Whether LLM explicitly signaled sub-goal completion via tool output",
    )
    rationale_verified: bool = Field(
        default=False,
        description="Whether LLM rationale contains sub-goal completion keywords",
    )
    trace_verified: bool = Field(
        default=False,
        description="Whether action trace confirms sub-goal completion",
    )
    screen_verified: bool = Field(
        default=False,
        description="Whether the screen changed after action execution, confirming the action had effect",
    )

    model_config = ConfigDict(frozen=True)

    def calculate_confidence(self) -> float:
        """
        Calculate weighted confidence score from all signals.

        Uses same weighting as CompletionSignal:
        - LLM confidence >= 0.8: +0.5
        - LLM confidence >= 0.5: +0.25
        - Keyword match: +0.3
        - Action executed (with screen change): +0.15
        - LLM signaled: +0.2 (explicit signal)
        - Trace verified: +0.2
        - Rationale verified: +0.1
        """
        score = 0.0

        # LLM confidence contribution
        if self.llm_confidence >= 0.8:
            score += 0.5
        elif self.llm_confidence >= 0.5:
            score += 0.25

        # Signal contributions
        if self.keyword_match:
            score += 0.3
        if self.action_executed and self.screen_verified:
            score += 0.15
        if self.llm_signaled:
            score += 0.2
        if self.trace_verified:
            score += 0.2
        if self.rationale_verified:
            score += 0.1

        return min(score, 1.0)

    def count_signals(self) -> int:
        """
        Count how many verification signals are present.

        Returns:
            Number of True boolean signals used by the completion gate.
            Policy: llm_signaled + rationale_verified + effective_action.
            ``action_executed`` is only counted when ``screen_verified`` is
            also True — this prevents premature sub-goal advancement when an
            action fires but the screen does not actually change (e.g. tap
            failed, blocking overlay, wrong screen).
        """
        effective_action = self.action_executed and self.screen_verified
        return sum(
            [
                self.llm_signaled,
                self.rationale_verified,
                effective_action,
            ]
        )

    def meets_threshold(self, required_signals: int = 2) -> bool:
        """
        Check if enough signals are present for completion gate.

        Args:
            required_signals: Minimum number of signals required (default: 2)

        Returns:
            True if threshold met
        """
        return self.count_signals() >= required_signals


class CompletionSignal(BaseModel):
    """
    Signals that indicate intent completion.
    Captures evidence that the goal has been achieved.
    """

    evidence: str = Field(default="", description="Textual evidence of completion")
    llm_confidence: float = Field(default=0.0, description="Confidence score from LLM")

    keyword_match: bool = Field(default=False, description="Whether keywords matched")
    expected_screen: bool = Field(default=False, description="Whether LLM predicted completion")
    success_indicator: bool = Field(default=False, description="Whether success patterns matched")
    llm_signaled: bool = Field(
        default=False,
        description="Whether LLM/action emitted explicit completion signal",
    )
    rationale_verified: bool = Field(
        default=False,
        description="Whether action/reasoning rationale verifies completion",
    )
    trace_verified: bool = Field(
        default=False,
        description="Whether trace/screen evidence verified completion",
    )

    model_config = ConfigDict(frozen=True)

    @property
    def is_complete(self) -> bool:
        """
        Determine if signals indicate completion.

        Uses weighted scoring:
        - LLM confidence > 0.8: high weight
        - Keyword match: medium weight
        - Success indicator: medium weight
        - Expected screen: low weight (may be intermediate)
        """

        score = 0.0

        if self.llm_confidence >= 0.8:
            score += 0.5

        elif self.llm_confidence >= 0.5:
            score += 0.25

        if self.keyword_match:
            score += 0.3

        if self.success_indicator:
            score += 0.15

        if self.expected_screen:
            score += 0.1

        return score >= 0.5
