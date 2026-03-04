from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class GeminiDeltaSignal(BaseModel):
    """
    Optional model-provided hints describing screen-level semantic deltas.
    """

    previous_screen_summary: Optional[str] = Field(default=None)
    current_screen_summary: Optional[str] = Field(default=None)
    delta_observed: Optional[bool] = Field(default=None)
    delta_reasoning: Optional[str] = Field(default=None)
    delta_confidence: Optional[float] = Field(default=None)
    visible_anchors: List[str] = Field(default_factory=list)
    top_anchor: Optional[str] = Field(default=None)
    bottom_anchor: Optional[str] = Field(default=None)
