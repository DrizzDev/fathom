from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


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

        if self.success_indicator:
            return True

        score = 0.0

        if self.llm_confidence >= 0.8:
            score += 0.5
        elif self.llm_confidence >= 0.5:
            score += 0.25

        if self.keyword_match:
            score += 0.3
        if self.expected_screen:
            score += 0.1

        return score >= 0.5
