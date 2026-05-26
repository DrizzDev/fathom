from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class SubGoalCompletionSignal(BaseModel):
    """
    Multi-signal verification for sub-goal completion.
    """

    evidence: str = Field(default="", description="Textual evidence of sub-goal completion")
    llm_confidence: float = Field(default=0.0, description="Confidence score from LLM")
    keyword_match: bool = Field(default=False, description="Sub-goal keywords matched reasoning")
    action_executed: bool = Field(
        default=False, description="An action was executed for this sub-goal"
    )
    flagged_complete: bool = Field(
        default=False, description="Model raised the completion flag via tool output"
    )
    rationale_verified: bool = Field(
        default=False, description="Model rationale matches sub-goal keywords"
    )
    trace_verified: bool = Field(
        default=False, description="Action trace confirms sub-goal completion"
    )
    screen_verified: bool = Field(
        default=False, description="Post-action screen change exceeded the meaningful-delta floor"
    )

    model_config = ConfigDict(frozen=True)

    def calculate_confidence(self) -> float:
        """
        Weighted confidence score from all signals.
        """

        effective_action = self.action_executed and self.screen_verified
        score = 0.0

        if self.llm_confidence >= 0.8:
            score += 0.5

        elif self.llm_confidence >= 0.5:
            score += 0.25

        if self.keyword_match:
            score += 0.3

        if effective_action:
            score += 0.15

        if self.flagged_complete:
            score += 0.2

        if self.trace_verified:
            score += 0.2

        if self.rationale_verified:
            score += 0.1

        return min(score, 1.0)

    def count_signals(self) -> int:
        """
        Count independent positive completion signals under the strict policy.

        Policy mirrors the ``main``-branch gate: ``flagged_complete +
        rationale_verified + effective_action``, where ``effective_action``
        requires both an executed action and a verified post-action screen
        change. ``trace_verified`` is intentionally excluded — it produced
        false positives historically and is disabled at the agent_state
        marking layer.
        """

        effective_action = self.action_executed and self.screen_verified
        return sum(
            (
                self.flagged_complete,
                self.rationale_verified,
                effective_action,
            )
        )

    def meets_threshold(self, *, required: int) -> bool:
        """
        Return True when enough independent positive signals agree to advance the sub-goal.
        """

        return self.count_signals() >= required


class CompletionSignal(BaseModel):
    """
    Signals that indicate intent completion.
    """

    evidence: str = Field(default="", description="Textual evidence of completion")
    llm_confidence: float = Field(default=0.0, description="Confidence score from LLM")
    keyword_match: bool = Field(default=False, description="Whether keywords matched")
    expected_screen: bool = Field(default=False, description="Whether LLM predicted completion")
    success_indicator: bool = Field(default=False, description="Whether success patterns matched")
    flagged_complete: bool = Field(
        default=False, description="Model raised an explicit completion flag"
    )
    rationale_verified: bool = Field(
        default=False, description="Action/reasoning rationale verifies completion"
    )
    trace_verified: bool = Field(
        default=False, description="Trace/screen evidence verified completion"
    )

    model_config = ConfigDict(frozen=True)

    @property
    def is_complete(self) -> bool:
        """
        Whether the signals' weighted score indicates completion.
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
