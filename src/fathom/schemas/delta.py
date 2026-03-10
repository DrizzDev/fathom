from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class GeminiDeltaSignal(BaseModel):
    """
    Normalized semantic delta signal used by the planner and agent state.

    Design goals:
    - Preserve raw provider values for auditability and analytics.
    - Expose normalized fields for downstream decision-making.
    - Distinguish between:
        * "Model says delta"      → delta_observed is True/False
        * "Model is unsure/low confidence" via delta_confidence
        * "Model said nothing"    → this object may be omitted entirely
    """

    # High-level semantic summaries
    previous_screen_summary: Optional[str] = Field(default=None)
    current_screen_summary: Optional[str] = Field(default=None)

    # Normalized primary signals used by downstream logic.
    # NOTE: These should be treated as the single source of truth for planning.
    delta_observed: Optional[bool] = Field(
        default=None,
        description="Normalized boolean signal indicating whether a meaningful screen change was observed.",
    )
    delta_confidence: Optional[float] = Field(
        default=None,
        description="Normalized confidence score in [0.0, 1.0] associated with delta_observed.",
    )

    # Optional free-form reasoning from the model.
    delta_reasoning: Optional[str] = Field(default=None)

    # Raw provider values preserved for audit and debugging.
    raw_delta_observed: Optional[bool] = Field(
        default=None,
        description="Raw provider value for delta_observed before normalization.",
    )
    raw_delta_confidence: Optional[float] = Field(
        default=None,
        description="Raw provider value for delta_confidence before normalization or clamping.",
    )

    # Metadata describing how normalization was applied.
    confidence_source: Optional[str] = Field(
        default=None,
        description="Origin of delta_confidence: 'model' (direct), 'system_clamped' (out-of-range corrected), or 'system_default' (fabricated fallback).",
    )
    observed_source: Optional[str] = Field(
        default=None,
        description="Origin of delta_observed when it differs from raw_delta_observed.",
    )

    # Anchor-based localization hints
    visible_anchors: List[str] = Field(default_factory=list)
    top_anchor: Optional[str] = Field(default=None)
    bottom_anchor: Optional[str] = Field(default=None)
