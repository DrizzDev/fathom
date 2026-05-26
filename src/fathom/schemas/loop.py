from __future__ import annotations

from enum import StrEnum
from typing import Tuple

from pydantic import BaseModel, ConfigDict, Field

from fathom.schemas.effect import ActionEffectStatus
from fathom.schemas.vision import ActionKind


class LoopReason(StrEnum):
    """
    Identifies which :class:`LoopDetector` strategy classified the recent window as stuck.

    The detector is multi-strategy; the reason exposes which strategy fired so the
    HITL policy and telemetry can interpret evidence without re-deriving it.
    """

    NOT_STUCK = "not_stuck"
    INERT_REPETITION = "inert_repetition"
    SCREEN_REPETITION = "screen_repetition"
    NEAR_DUPLICATE_VISUAL = "near_duplicate_visual"
    STATE_OSCILLATION = "state_oscillation"
    SCROLL_STALL = "scroll_stall"
    ACTION_VELOCITY = "action_velocity"
    ACTION_REPETITION = "action_repetition"


class LoopTurn(BaseModel):
    """
    Read-only snapshot of one turn recorded by the loop detector, kept narrow
    enough that the HITL policy can decide without leaking detector internals.
    """

    model_config = ConfigDict(frozen=True)

    action_kind: ActionKind = Field(
        description="Functional categorization of the action recorded for this turn."
    )
    action_type: str = Field(
        description="Raw action_type token recorded by the detector (lower-case).",
    )
    effect_status: ActionEffectStatus = Field(
        description="Post-action effect classification (PROGRESS / NO_PROGRESS / UNCERTAIN).",
    )
    screen_hash_prefix: str = Field(
        default="",
        description="Short visual-hash prefix for the turn's screen; diagnostic only.",
    )


class LoopEvidence(BaseModel):
    """
    Typed evidence snapshot returned by :meth:`LoopDetector.evidence`.

    ``recent`` is the full window for telemetry. ``since_progress`` is the
    trailing contiguous turns starting after the most recent PROGRESS effect,
    computed by walking the window. The escalation gate must only consider
    ``since_progress`` — that is the load-bearing distinction that prevents
    historical, unrelated no-progress turns from unlocking escalation.
    """

    model_config = ConfigDict(frozen=True)

    stuck: bool = Field(
        description="True when the loop detector classifies the current window as stuck.",
    )
    reason: LoopReason = Field(
        description="Which detection strategy fired; NOT_STUCK when ``stuck`` is False.",
    )
    recent: Tuple[LoopTurn, ...] = Field(
        default_factory=tuple,
        description="Full window snapshot, oldest-first; diagnostic / telemetry context.",
    )
    since_progress: Tuple[LoopTurn, ...] = Field(
        default_factory=tuple,
        description=(
            "Trailing contiguous turns since the most recent PROGRESS effect. "
            "Bounds the slice that the escalation gate is allowed to consider; "
            "anything earlier belongs to a prior recovery cycle. UNCERTAIN turns "
            "are included as-is."
        ),
    )
