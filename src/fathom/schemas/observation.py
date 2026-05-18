from __future__ import annotations

from enum import StrEnum
from typing import List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field

from fathom.schemas.actions import Bounds
from fathom.schemas.artifacts import StepArtifacts
from fathom.schemas.screens import ScreenDiff, ScreenHashBundle


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
    OSCILLATING = "oscillating"
    NEAR_DUPLICATE = "near_duplicate"


class LoopObservation(BaseModel):
    """
    Structured loop-observation handed to the agent through the
    ANALYZE prompt when the deterministic LoopDetector has accumulated
    enough evidence to call the current sub-goal stuck.

    **Information, not control.** The observation is rendered into the
    prompt as a system note. The agent decides whether to try a
    different target, call ``ask_user``, or call ``request_replan``
    with a typed :class:`fathom.schemas.escape.EscapeCategory`. The
    system does not override the agent's next action.

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
            "request_replan with category=target_not_available if no "
            "alternative target makes sense'). Never an instruction."
        ),
    )


class ElementSource(StrEnum):
    """
    Source that produced a perceived element.
    """

    XML = "xml"
    OCR = "ocr"
    CV = "cv"
    ICON = "icon"
    MODEL = "model"
    VISION = "vision"
    ACCESSIBILITY = "accessibility"


class ElementRole(StrEnum):
    """
    Semantic role assigned to a perceived element.
    """

    ICON = "icon"
    TEXT = "text"
    INPUT = "input"
    BUTTON = "button"
    OVERLAY = "overlay"
    UNKNOWN = "unknown"
    KEYBOARD = "keyboard"
    CONTAINER = "container"
    SCROLL_REGION = "scroll_region"


class PerceivedElement(BaseModel):
    """
    Unified screen element produced by perception providers.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    identifier: str = Field(description="Stable identifier within the current observation.")

    bounds: Bounds = Field(description="Element bounds in screen coordinates.")
    source: ElementSource = Field(description="Provider that produced the element.")

    role: ElementRole = Field(description="Best known semantic role.")
    confidence: float = Field(ge=0.0, le=1.0, description="Provider confidence.")
    text: Optional[str] = Field(default=None, description="Visible or accessibility text.")
    tappable: bool = Field(description="Whether the element can be used as an action target.")

    parent: Optional[str] = Field(default=None, description="Parent element identifier.")
    label_id: Optional[str] = Field(
        default=None,
        description=(
            "Numeric label drawn on annotated artifacts. Mirrors the drawer's "
            "manifest label so all rendered perception images use the same "
            "numeric anchor (``[N]``) the planner sees in the manifest."
        ),
    )


class KeyboardObservation(BaseModel):
    """
    Keyboard state detected on the screen.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    visible: bool = Field(description="Whether a software keyboard is visible.")
    bounds: Optional[Bounds] = Field(default=None, description="Keyboard bounds when visible.")

    dismiss: Tuple[PerceivedElement, ...] = Field(
        default_factory=tuple,
        description="Known controls or safe actions that can dismiss the keyboard.",
    )


class OverlayObservation(BaseModel):
    """
    Blocking overlay detected on the screen.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    bounds: Bounds = Field(description="Overlay bounds.")
    visible: bool = Field(description="Whether a blocking overlay is present.")

    candidates: Tuple[PerceivedElement, ...] = Field(
        default_factory=tuple,
        description="Dismiss candidates such as primary buttons or close icons.",
    )


class ScrollRegion(BaseModel):
    """
    Scrollable region candidate detected on the screen.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    bounds: Bounds = Field(description="Region bounds.")
    direction: str = Field(description="Supported scroll direction.")
    confidence: float = Field(ge=0.0, le=1.0, description="Provider confidence.")


class ScreenObservation(BaseModel):
    """
    Unified screen observation used by decision, localization, and supervision.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    activity: str = Field(description="Current application activity or package context.")
    elements: Tuple[PerceivedElement, ...] = Field(description="Merged perceived elements.")
    hashes: ScreenHashBundle = Field(description="Visual and structural hashes for the screen.")

    overlays: Tuple[OverlayObservation, ...] = Field(
        default_factory=tuple,
        description="Blocking overlays visible on the screen.",
    )

    keyboard: KeyboardObservation = Field(description="Keyboard state.")

    scroll: Tuple[ScrollRegion, ...] = Field(
        default_factory=tuple,
        description="Scrollable region candidates.",
    )
    calls_to_action: Tuple[PerceivedElement, ...] = Field(
        default_factory=tuple,
        description="Visible terminal or prominent action controls.",
    )
    focused: Optional[PerceivedElement] = Field(
        default=None,
        description="Focused input or control when known.",
    )


class PostActionObservation(BaseModel):
    """
    Screen observation and comparison captured after one executed action.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    post_visual_hash: Optional[str] = Field(
        default=None,
        description="Visual hash of the post-action capture when available.",
    )
    screen_diff: Optional[ScreenDiff] = Field(
        default=None,
        description="Rich screen comparison between pre-action and post-action captures.",
    )
    observation: Optional[ScreenObservation] = Field(
        default=None,
        description="Post-action screen observation when capture succeeded.",
    )
    artifacts: Optional[StepArtifacts] = Field(
        default=None,
        description=(
            "Namespaced artifact envelope (screen.before / screen.after) "
            "persisted via the StoragePort during post-action capture."
        ),
    )
