from __future__ import annotations

from enum import StrEnum
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class ScreenRelation(StrEnum):
    """
    How the recent screens relate to each other.

    - ``DIVERGING``: hashes are moving apart (productive scrolling /
      exploration). Loop observation should NOT fire.
    - ``NEAR_DUPLICATE``: hashes are clustered within the loop-detector
      hamming threshold (the agent is scrolling / tapping on a screen
      that is not responding).
    - ``OSCILLATING``: alternating between two distinct screens (e.g.
      an overlay that re-opens after each dismissal attempt).
    """

    DIVERGING = "diverging"
    NEAR_DUPLICATE = "near_duplicate"
    OSCILLATING = "oscillating"


class LoopObservation(BaseModel):
    """
    Structured loop-observation handed to the agent through the
    ANALYZE prompt when the deterministic LoopDetector has accumulated
    enough evidence to call the current sub-goal stuck.

    **Information, not control.** The observation is rendered into the
    prompt as a system note. The agent decides whether to try a
    different target, call ``ask_user``, or call
    ``report_screen_unactionable``. The system does not override the
    agent's next action.

    The schema is intentionally minimal — only what the agent needs to
    reason about its trajectory. Raw timestamps, full screen hashes,
    and other forensic detail stay in telemetry.
    """

    model_config = ConfigDict(frozen=True)

    repeated_action: str = Field(
        description=(
            "Most-repeated action descriptor in the recent window, "
            "e.g. 'Swipe up on Auto suggest page'."
        ),
    )
    count: int = Field(
        ge=2,
        description="How many times the repeated action has occurred in the recent window.",
    )
    progress_scores: List[float] = Field(
        default_factory=list,
        description=(
            "Visual-progress scores from the most recent actions, "
            "oldest first. Empty when action effect telemetry is unavailable."
        ),
    )
    screen_relation: ScreenRelation = Field(
        description="How the recent screens relate to each other.",
    )
    suggested_alternatives: List[str] = Field(
        default_factory=list,
        description=(
            "Optional list of manifest-visible alternative targets the "
            "agent might consider. Empty when no alternatives were "
            "available to surface."
        ),
    )
    note: Optional[str] = Field(
        default=None,
        description=(
            "Optional short, non-prescriptive note (e.g. 'consider "
            "report_screen_unactionable if no alternative target makes "
            "sense'). Never an instruction."
        ),
    )
