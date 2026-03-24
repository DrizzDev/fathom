from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class GeminiDeltaSignal(BaseModel):
    """Optional Gemini-provided signals for no-XML delta interpretation."""

    previous_screen_summary: Optional[str] = Field(default=None)
    current_screen_summary: Optional[str] = Field(default=None)
    delta_observed: Optional[bool] = Field(default=None)
    delta_reasoning: Optional[str] = Field(default=None)
    delta_confidence: Optional[float] = Field(default=None)
    visible_anchors: List[str] = Field(default_factory=list)
    top_anchor: Optional[str] = Field(default=None)
    bottom_anchor: Optional[str] = Field(default=None)


class ScreenDeltaSignal(BaseModel):
    """Deterministic no-XML screen delta computed by runtime."""

    no_xml: bool = Field(default=True)
    changed: bool = Field(default=False)
    delta_score: float = Field(default=0.0)
    reason: str = Field(default="unavailable")
    pre_activity: Optional[str] = Field(default=None)
    post_activity: Optional[str] = Field(default=None)
    pre_visual_hash: Optional[str] = Field(default=None)
    post_visual_hash: Optional[str] = Field(default=None)
    gemini: Optional[GeminiDeltaSignal] = Field(default=None)
