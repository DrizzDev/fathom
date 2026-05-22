from __future__ import annotations

from typing import Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field

from fathom.constants.scroll import (
    ScrollDirection,
    ScrollEvidenceSource,
    ScrollStage,
    ScrollVerdictKind,
    SurfaceKind,
)
from fathom.schemas.actions import Bounds, ExecutionRegion, GesturePath
from fathom.schemas.command import CommandAnchor, CommandScope


class ScrollScope(CommandScope):
    """
    Resolved execution scope for one scroll command.
    """

    source: ScrollEvidenceSource = Field(description="Evidence source that resolved this scope.")
    manifest_label_id: Optional[str] = Field(
        default=None,
        description="Resolved manifest label identifier when present.",
    )
    observation_region_id: Optional[str] = Field(
        default=None,
        description="Resolved observation-only region identifier when present.",
    )


class ScrollSurface(BaseModel):
    """
    Surface hint that may interfere with one scroll gesture.
    """

    model_config = ConfigDict(frozen=True)

    kind: SurfaceKind = Field(description="Class of interfering surface.")
    bounds: Bounds = Field(description="Capture-space bounds of the interfering surface.")
    source: ScrollEvidenceSource = Field(description="Evidence source for this surface hint.")
    detail: Optional[str] = Field(default=None, description="Short diagnostic note.")


class ScrollVerdict(BaseModel):
    """
    Deterministic assessment of one scroll attempt.
    """

    model_config = ConfigDict(frozen=True)

    kind: ScrollVerdictKind = Field(description="Observed scroll outcome class.")
    source: ScrollEvidenceSource = Field(description="Evidence source that produced the verdict.")
    confidence: float = Field(ge=0.0, le=1.0, description="Verdict strength in [0, 1].")
    distance: int = Field(
        ge=0,
        description="Observed translation magnitude along the dominant axis in capture pixels.",
    )
    detail: Optional[str] = Field(default=None, description="Short diagnostic note.")


class ScrollAttempt(BaseModel):
    """
    One dispatched scroll attempt.
    """

    model_config = ConfigDict(frozen=True)

    stage: ScrollStage = Field(description="Adaptive stage that produced the gesture.")
    path: GesturePath = Field(description="Gesture path dispatched to the device.")
    region: ExecutionRegion = Field(description="Logical execution region used to derive the path.")
    scope: ScrollScope = Field(description="Resolved scope that constrained this attempt.")
    capture_region: Bounds = Field(description="Capture-space region used for outcome observation.")
    avoided: Tuple[ScrollSurface, ...] = Field(
        default_factory=tuple,
        description="Surface hints the planner intentionally avoided for this attempt.",
    )
    verdict: Optional[ScrollVerdict] = Field(
        default=None,
        description="Observed verdict for this attempt when available.",
    )


class ScrollOutcome(BaseModel):
    """
    Final bounded outcome of one supervised scroll run.
    """

    model_config = ConfigDict(frozen=True)

    success: bool = Field(description="Whether the supervisor confirmed useful movement.")
    attempts: Tuple[ScrollAttempt, ...] = Field(
        default_factory=tuple,
        description="Ordered attempts performed for this scroll run.",
    )
    final: ScrollVerdict = Field(description="Final verdict returned to the caller.")
    scope: Optional[ScrollScope] = Field(
        default=None,
        description="Resolved scope used for the supervised scroll run.",
    )


class ScrollLock(BaseModel):
    """
    Stable scroll container lock carried across repeated scroll steps.
    """

    model_config = ConfigDict(frozen=True)

    scope: ScrollScope = Field(
        description="Resolved scope to reuse for the active scroll objective."
    )
    direction: ScrollDirection = Field(
        description="Locked content-movement direction for the active scroll objective."
    )
    target: str = Field(description="Normalized semantic target of the active scroll objective.")


class ScrollContext(BaseModel):
    """
    Input context for one adaptive scroll evaluation.
    """

    model_config = ConfigDict(frozen=True)

    direction: ScrollDirection = Field(description="Intended scroll direction.")
    region: ExecutionRegion = Field(description="Logical execution region of the original request.")
    anchor: CommandAnchor = Field(
        default_factory=CommandAnchor,
        description="Anchor carried from planner output into scope resolution.",
    )
