from __future__ import annotations

from typing import Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from fathom.constants.swipe import (
    DEFAULT_SWIPE_MINIMUM_TRAVEL,
    DEFAULT_SWIPE_MINIMUM_TRAVEL_FLOOR,
    DEFAULT_SWIPE_RETRY_DIRECTION,
    DEFAULT_SWIPE_RETRY_ENABLED,
    DEFAULT_SWIPE_RETRY_MAGNITUDES,
    AbortReason,
    RetryDirection,
)
from fathom.schemas.actions import GesturePath


class SwipeRetryPolicy(BaseModel):
    """
    Bounded coordinate-only retry policy for a swipe whose post-action capture matched the pre-action capture.
    """

    model_config = ConfigDict(frozen=True)

    enabled: bool = Field(
        default=DEFAULT_SWIPE_RETRY_ENABLED,
        description="When False the executor dispatches exactly one swipe with no retries.",
    )
    direction: RetryDirection = Field(
        default=DEFAULT_SWIPE_RETRY_DIRECTION,
        description="Direction the start point is shifted on each retry.",
    )
    magnitudes: Tuple[float, ...] = Field(
        default=DEFAULT_SWIPE_RETRY_MAGNITUDES,
        description="Per-retry shift sizes as fractions of the original gesture travel.",
    )
    minimum_travel: int = Field(
        default=DEFAULT_SWIPE_MINIMUM_TRAVEL,
        ge=DEFAULT_SWIPE_MINIMUM_TRAVEL_FLOOR,
        description="Lower bound on the post-shift gesture travel in pixels.",
    )

    @model_validator(mode="after")
    def __magnitudes_well_formed(self) -> "SwipeRetryPolicy":
        """
        Reject zero-length, non-monotonic, or out-of-range magnitudes when retry is enabled.
        """

        if not self.enabled:
            return self

        if len(self.magnitudes) == 0:
            raise ValueError("magnitudes must contain at least one fraction when retry is enabled")

        for magnitude in self.magnitudes:
            if not (0.0 < magnitude <= 0.5):
                raise ValueError(f"magnitude {magnitude} outside (0.0, 0.5]")

        return self


class DeviceOutcome(BaseModel):
    """
    Adapter-level dispatch outcome for one swipe attempt.
    """

    model_config = ConfigDict(frozen=True)

    succeeded: bool = Field(description="Device adapter reported success for the swipe primitive.")
    error: Optional[str] = Field(
        default=None, description="Device error message when succeeded is False."
    )


class VisualOutcome(BaseModel):
    """
    Pre/post visual-hash comparison for one swipe attempt.
    """

    model_config = ConfigDict(frozen=True)

    changed: bool = Field(
        description="After hash is present AND differs from the original before hash."
    )
    before: str = Field(description="Visual hash of the original pre-action capture.")
    after: Optional[str] = Field(
        default=None, description="Visual hash post-attempt; None on capture failure."
    )


class SwipeAttempt(BaseModel):
    """
    One dispatched swipe attempt with its observed effect relative to the original pre-action capture.
    """

    model_config = ConfigDict(frozen=True)

    index: int = Field(ge=0, description="Zero-based attempt index within the execution sequence.")
    path: GesturePath = Field(description="Path dispatched to the device.")
    device: DeviceOutcome = Field(description="Adapter-level dispatch outcome.")
    visual: VisualOutcome = Field(description="Pre/post visual-hash comparison outcome.")


class SwipeRejection(BaseModel):
    """
    A retry candidate that was filtered before dispatch.
    """

    model_config = ConfigDict(frozen=True)

    index: int = Field(ge=0, description="Position in the original ordered candidate sequence.")
    path: GesturePath = Field(description="Candidate path that was not dispatched.")
    reason: AbortReason = Field(description="Why this candidate was rejected.")


class CandidateSequence(BaseModel):
    """
    Planner output: ordered accepted candidates plus filtered rejections.
    """

    model_config = ConfigDict(frozen=True)

    accepted: Tuple[GesturePath, ...] = Field(
        default_factory=tuple,
        description="Candidates that passed all safety filters, in dispatch order.",
    )
    rejections: Tuple[SwipeRejection, ...] = Field(
        default_factory=tuple,
        description="Candidates that did not pass filters, with reason.",
    )


class SwipeExecution(BaseModel):
    """
    Aggregated outcome of one logical swipe action including any bounded retries.
    """

    model_config = ConfigDict(frozen=True)

    attempts: Tuple[SwipeAttempt, ...] = Field(
        default_factory=tuple,
        description="Attempts actually dispatched to the device, in order.",
    )
    rejections: Tuple[SwipeRejection, ...] = Field(
        default_factory=tuple,
        description="Candidates filtered before dispatch, in candidate order.",
    )
    final: Optional[GesturePath] = Field(
        default=None,
        description="Path of the last attempt dispatched; None when no attempt was dispatched.",
    )
    aborted_for: Optional[AbortReason] = Field(
        default=None,
        description="Resolved abort reason when execution did not produce an effective scroll.",
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def any_dispatched(self) -> bool:
        """
        Whether at least one attempt was dispatched to the device.
        """

        return len(self.attempts) > 0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def any_visual_change(self) -> bool:
        """
        Whether at least one attempt's after-hash differed from the original before-hash.
        """

        return any(attempt.visual.changed for attempt in self.attempts)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def effective(self) -> bool:
        """
        Whether at least one attempt's device dispatch succeeded AND its visual outcome changed.
        """

        return any(attempt.device.succeeded and attempt.visual.changed for attempt in self.attempts)
