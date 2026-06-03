from __future__ import annotations

from enum import StrEnum
from typing import Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, model_validator

from fathom.constants.localization import LocalizationGridScale
from fathom.schemas.actions import Bounds
from fathom.schemas.observation import ElementSource, PerceivedElement


class VisionLocalizationPayload(BaseModel):
    """
    Vision localizer response — bounding rectangle on the normalized integer grid.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    x1: int = Field(
        ge=LocalizationGridScale.MINIMUM,
        le=LocalizationGridScale.MAXIMUM,
        description="Left edge of the target bounding rectangle.",
    )
    y1: int = Field(
        ge=LocalizationGridScale.MINIMUM,
        le=LocalizationGridScale.MAXIMUM,
        description="Top edge of the target bounding rectangle.",
    )
    x2: int = Field(
        ge=LocalizationGridScale.MINIMUM,
        le=LocalizationGridScale.MAXIMUM,
        description="Right edge of the target bounding rectangle.",
    )
    y2: int = Field(
        ge=LocalizationGridScale.MINIMUM,
        le=LocalizationGridScale.MAXIMUM,
        description="Bottom edge of the target bounding rectangle.",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Model self-reported confidence for the proposed bound.",
    )
    rationale: str = Field(description="One-sentence justification for the bound.")

    @property
    def refused(self) -> bool:
        """
        Whether this payload signals the localizer's refusal protocol.
        """

        return (
            self.x1 == LocalizationGridScale.MINIMUM
            and self.y1 == LocalizationGridScale.MINIMUM
            and self.x2 == LocalizationGridScale.MINIMUM
            and self.y2 == LocalizationGridScale.MINIMUM
            and self.confidence == 0.0
        )

    @model_validator(mode="after")
    def __check_axes(self) -> "VisionLocalizationPayload":
        """
        Reject non-refusal payloads with inverted axes or zero area.
        """

        if self.refused:
            return self
        if self.x1 >= self.x2 or self.y1 >= self.y2:
            raise ValueError("axes inverted or zero area")
        return self


class Point(BaseModel):
    """
    Absolute screen point used for action execution.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    x: int = Field(ge=0, description="Horizontal coordinate in screen pixels.")
    y: int = Field(ge=0, description="Vertical coordinate in screen pixels.")


class LocalizationStatus(StrEnum):
    """
    Target localization result states.
    """

    RESOLVED = "resolved"
    AMBIGUOUS = "ambiguous"
    UNRESOLVED = "unresolved"


class EnsembleMemberName(StrEnum):
    """
    Stable names for ensemble vision-localizer members.
    """

    GEMINI_VISION = "gemini_vision"
    DOCUMENT_AI_LAYOUT = "document_ai_layout"


class LocalizationCandidate(BaseModel):
    """
    Candidate target returned by localization.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    reason: str = Field(description="Reason the candidate matches.")
    score: float = Field(ge=0.0, le=1.0, description="Candidate match score.")
    point: Optional[Point] = Field(default=None, description="Candidate action point.")
    element: Optional[PerceivedElement] = Field(default=None, description="Matched element.")


class LocalizationProposal(BaseModel):
    """
    One ensemble-member proposal for a semantic target's bounding box.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    bounds: Bounds = Field(description="Pixel bounds of the proposed match.")
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Member-reported confidence in the closed unit interval.",
    )
    rationale: Optional[str] = Field(default=None, description="Optional human-readable reason.")
    source: str = Field(min_length=1, description="Stable name of the proposing localizer member.")


class LocalizationResult(BaseModel):
    """
    Final localization result for a semantic action target.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: LocalizationStatus = Field(description="Localization outcome.")
    point: Optional[Point] = Field(default=None, description="Resolved action point.")
    bounds: Optional[Bounds] = Field(default=None, description="Resolved action bounds.")
    source: Optional[ElementSource] = Field(default=None, description="Source used for resolution.")

    confidence: float = Field(ge=0.0, le=1.0, description="Final localization confidence.")
    candidates: Tuple[LocalizationCandidate, ...] = Field(
        default_factory=tuple,
        description="Candidate targets when localization is ambiguous or unresolved.",
    )

    reason: Optional[str] = Field(default=None, description="Diagnostic result reason.")
