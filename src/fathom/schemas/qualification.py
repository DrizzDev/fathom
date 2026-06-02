from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from fathom.constants.qualification import QualificationLabel, RationaleCategory


class Rationale(BaseModel):
    """
    Structured rationale carried by a qualification verdict.
    """

    model_config = ConfigDict(frozen=True)

    reasoning: str = Field(default="", description="One-sentence rationale.")
    category: RationaleCategory = Field(description="Reasoning bucket for the verdict.")


class QualificationVerdict(BaseModel):
    """
    Structured verdict produced by an intent qualifier.
    """

    model_config = ConfigDict(frozen=True)

    label: QualificationLabel = Field(description="Executability classification.")
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Model confidence in the chosen label.",
    )
    rationale: Rationale = Field(description="Structured reasoning behind the label.")
    message: Optional[str] = Field(
        default=None,
        description="Friendly explanation surfaced to the user (set only when blocking).",
    )

    def should_block(self, *, floor: float) -> bool:
        """
        Whether this verdict blocks execution given the configured confidence floor.
        """

        return self.label == QualificationLabel.NOT_EXECUTABLE and self.confidence >= floor
