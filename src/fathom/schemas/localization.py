from __future__ import annotations

from enum import StrEnum
from typing import Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, model_validator

from fathom.constants.localization import (
    LAYOUT_MAX_HEIGHT_RATIO,
    LAYOUT_MAX_HORIZONTAL_GAP_RATIO,
    LAYOUT_MAX_ROW_OFFSET_RATIO,
    LAYOUT_MIN_TOKEN_CONFIDENCE,
    LAYOUT_MIN_WORD_LENGTH_FOR_FUZZ,
    LAYOUT_PER_WORD_SIMILARITY_THRESHOLD,
    LAYOUT_PHRASE_MATCH_THRESHOLD,
    LocalizationGridScale,
)
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


class PhraseMatch(BaseModel):
    """
    Merged phrase formed by clustering adjacent OCR tokens for a single match.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    bounds: Bounds = Field(description="Union pixel bounds of the clustered tokens.")
    text: str = Field(min_length=1, description="Concatenated phrase text from the cluster.")

    score: float = Field(
        ge=0.0,
        le=1.0,
        description="Match score between the target and this phrase on the unit interval.",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Lowest provider-reported token confidence across the cluster.",
    )
    token_count: int = Field(gt=0, description="Number of source tokens merged into this phrase.")


class LayoutMatchConfiguration(BaseModel):
    """
    Tunable's governing phrase clustering and target matching in the layout localizer.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    phrase_match_threshold: float = Field(
        ge=0.0,
        le=1.0,
        default=LAYOUT_PHRASE_MATCH_THRESHOLD,
        description="Minimum F1 score required to accept a phrase as a match.",
    )
    per_word_similarity_threshold: float = Field(
        ge=0.0,
        le=1.0,
        default=LAYOUT_PER_WORD_SIMILARITY_THRESHOLD,
        description="Per-word similarity ratio that treats two words as equivalent.",
    )
    min_word_length_for_fuzz: int = Field(
        gt=0,
        default=LAYOUT_MIN_WORD_LENGTH_FOR_FUZZ,
        description="Words shorter than this require exact equality instead of fuzzy similarity.",
    )
    min_token_confidence: float = Field(
        ge=0.0,
        le=1.0,
        default=LAYOUT_MIN_TOKEN_CONFIDENCE,
        description="Tokens below this provider confidence are excluded from clustering.",
    )
    max_row_offset_ratio: float = Field(
        gt=0.0,
        default=LAYOUT_MAX_ROW_OFFSET_RATIO,
        description="Maximum y-centre offset relative to token height for same-row clustering.",
    )
    max_horizontal_gap_ratio: float = Field(
        gt=0.0,
        default=LAYOUT_MAX_HORIZONTAL_GAP_RATIO,
        description="Maximum horizontal gap relative to token height for adjacency.",
    )
    max_height_ratio: float = Field(
        gt=1.0,
        default=LAYOUT_MAX_HEIGHT_RATIO,
        description="Maximum height ratio between adjacent tokens to remain in the same cluster.",
    )


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
