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
    RegionalEvidenceDecision,
)
from fathom.constants.perception import (
    CONTAINMENT_MINIMUM_RATIO,
    FUSED_WEIGHT_CONTAINMENT,
    FUSED_WEIGHT_DENSITY,
    FUSED_WEIGHT_IOU,
    FUSED_WEIGHT_RECALL,
    FUSED_WEIGHT_SUM_TOLERANCE,
    MODEL_BOUNDS_MINIMUM_IOU,
    PHRASE_DENSITY_FLOOR,
    PHRASE_MATCH_MINIMUM_RECALL,
    REGIONAL_EVIDENCE_FLOOR,
    VISION_LOCALIZER_ATTEMPTS,
    VISION_LOCALIZER_RETRY_BACKOFF,
    VISION_LOCALIZER_TIMEOUT,
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
            self.confidence == 0.0
            and self.x1 == LocalizationGridScale.MINIMUM
            and self.y1 == LocalizationGridScale.MINIMUM
            and self.x2 == LocalizationGridScale.MINIMUM
            and self.y2 == LocalizationGridScale.MINIMUM
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


class FusedScoreWeights(BaseModel):
    """
    Convex weights blending recall, density, containment, and IoU into the
    fused regional-evidence score. Must sum to 1.0.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    recall: float = Field(
        ge=0.0,
        le=1.0,
        default_factory=lambda: FUSED_WEIGHT_RECALL,
        description="Weight applied to target-word recall.",
    )
    density: float = Field(
        ge=0.0,
        le=1.0,
        default_factory=lambda: FUSED_WEIGHT_DENSITY,
        description="Weight applied to phrase-word density of target words.",
    )
    containment: float = Field(
        ge=0.0,
        le=1.0,
        default_factory=lambda: FUSED_WEIGHT_CONTAINMENT,
        description="Weight applied to best per-element containment ratio.",
    )
    iou: float = Field(
        ge=0.0,
        le=1.0,
        default_factory=lambda: FUSED_WEIGHT_IOU,
        description="Weight applied to best per-element IoU.",
    )

    @model_validator(mode="after")
    def __check_convex(self) -> "FusedScoreWeights":
        """
        Reject weight sets that do not sum to one within floating tolerance.
        """

        total = self.recall + self.density + self.containment + self.iou

        if abs(total - 1.0) > FUSED_WEIGHT_SUM_TOLERANCE:
            raise ValueError(f"Fused-score weights must sum to 1.0; got {total:.6f}.")

        return self


class RegionalEvidenceConfiguration(BaseModel):
    """
    Thresholds for ``RegionalEvidenceMatcher`` — recall, density, containment,
    and the IoU floor that decides whether LLM bounds + OCR phrase evidence
    inside those bounds is strong enough to dispatch the action.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    recall: float = Field(
        ge=0.0,
        le=1.0,
        default_factory=lambda: PHRASE_MATCH_MINIMUM_RECALL,
        description="Minimum fraction of target words that must appear in the matched phrase.",
    )
    density: float = Field(
        ge=0.0,
        le=1.0,
        default_factory=lambda: PHRASE_DENSITY_FLOOR,
        description="Minimum fraction of matched-phrase words that must come from the target.",
    )
    containment: float = Field(
        ge=0.0,
        le=1.0,
        default_factory=lambda: CONTAINMENT_MINIMUM_RATIO,
        description="Minimum fraction of an element's area that must lie inside the model bounds.",
    )
    iou: float = Field(
        ge=0.0,
        le=1.0,
        default_factory=lambda: MODEL_BOUNDS_MINIMUM_IOU,
        description="Symmetric-case IoU floor between the model bounds and one element.",
    )
    floor: float = Field(
        ge=0.0,
        le=1.0,
        default_factory=lambda: REGIONAL_EVIDENCE_FLOOR,
        description="Fused-score floor below which a proposal is rejected even with one strong signal.",
    )

    weights: FusedScoreWeights = Field(
        default_factory=FusedScoreWeights,
        description="Convex weights that combine recall, density, containment, and IoU.",
    )


class VisionRetryPolicy(BaseModel):
    """
    Retry policy for the vision localization adapter.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    attempts: int = Field(
        ge=1,
        default_factory=lambda: VISION_LOCALIZER_ATTEMPTS,
        description="Total attempts including the first call.",
    )
    backoff: float = Field(
        ge=1.0,
        default_factory=lambda: VISION_LOCALIZER_RETRY_BACKOFF,
        description="Multiplier applied between successive attempts.",
    )


class VisionLocalizationConfiguration(BaseModel):
    """
    Boot-time configuration for the vision localization adapter.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    timeout: int = Field(
        ge=1,
        default_factory=lambda: VISION_LOCALIZER_TIMEOUT,
        description="Per-attempt wait window in milliseconds.",
    )
    retry: VisionRetryPolicy = Field(
        default_factory=VisionRetryPolicy,
        description="Retry policy applied when an attempt times out.",
    )


class RegionalEvidenceMetrics(BaseModel):
    """
    Per-evaluation observability metrics for the regional matcher.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    recall: float = Field(ge=0.0, le=1.0, description="Target-word recall inside the cluster.")
    density: float = Field(ge=0.0, le=1.0, description="Cluster-word density of target words.")

    containment: float = Field(
        ge=0.0,
        le=1.0,
        description="Best per-element containment ratio inside the model bounds.",
    )
    iou: float = Field(
        ge=0.0,
        le=1.0,
        description="Best per-element IoU with the model bounds.",
    )
    fused: float = Field(
        ge=0.0,
        le=1.0,
        description="Weighted combination of recall, density, containment and IoU.",
    )

    @classmethod
    def zero(cls) -> "RegionalEvidenceMetrics":
        """
        Return zero-valued metrics for decisions where no math was computed.
        """

        return cls(recall=0.0, density=0.0, containment=0.0, iou=0.0, fused=0.0)


class RegionalEvidenceProposal(BaseModel):
    """
    Outcome of scoring perceived OCR evidence against planner-emitted bounds.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    bounds: Bounds = Field(description="Tight bounds covering the matched OCR phrase cluster.")
    score: float = Field(
        ge=0.0,
        le=1.0,
        description="Fused score over recall, density, containment, and IoU.",
    )
    recall: float = Field(
        ge=0.0, le=1.0, description="Target-word recall inside the matched phrase."
    )
    density: float = Field(
        ge=0.0,
        le=1.0,
        description="Matched-phrase density of target words.",
    )
    containment: float = Field(
        ge=0.0,
        le=1.0,
        description="Best per-element containment ratio inside the model bounds.",
    )
    iou: float = Field(
        ge=0.0,
        le=1.0,
        description="Best per-element IoU with the model bounds.",
    )
    phrase: str = Field(
        min_length=1,
        description="Concatenated phrase text from the chosen cluster.",
    )


class RegionalEvidenceVerdict(BaseModel):
    """
    Structured outcome of one ``RegionalEvidenceMatcher.evaluate`` call.
    Carries the decision, the math, and the proposal so callers can log
    every branch (resolved or rejected) with full attribution.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    metrics: RegionalEvidenceMetrics = Field(description="Per-gate metrics computed this turn.")
    decision: RegionalEvidenceDecision = Field(description="Why the matcher resolved or abstained.")

    proposal: Optional[RegionalEvidenceProposal] = Field(
        default=None,
        description="Tight tap-target proposal when the decision is RESOLVED.",
    )
    phrase: Optional[str] = Field(
        default=None,
        description="Cluster phrase text considered this turn, when a cluster was formed.",
    )
    cluster_token_count: int = Field(
        ge=0,
        description="Tokens merged into the considered cluster.",
    )
    in_region_token_count: int = Field(
        ge=0,
        description="OCR tokens whose centroid lay inside the model bounds.",
    )

    @property
    def resolved(self) -> bool:
        """
        Convenience flag for callers that only need the boolean outcome.
        """

        return self.decision is RegionalEvidenceDecision.RESOLVED


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
