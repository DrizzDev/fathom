from __future__ import annotations

from enum import StrEnum
from typing import Final


class RetryDirection(StrEnum):
    """
    Shift direction applied to the start point on each retry.
    """

    INWARD = "INWARD"
    OUTWARD = "OUTWARD"
    BOTH = "BOTH"


class AbortReason(StrEnum):
    """
    Reason a bounded swipe execution terminated without an effective scroll.
    """

    DEVICE_FAILED = "DEVICE_FAILED"
    CAPTURE_FAILED = "CAPTURE_FAILED"
    NO_VISUAL_CHANGE = "NO_VISUAL_CHANGE"
    KEYBOARD_BLOCKED = "KEYBOARD_BLOCKED"
    OUT_OF_BOUNDS = "OUT_OF_BOUNDS"
    UNSAFE_ANCHOR = "UNSAFE_ANCHOR"
    MINIMUM_TRAVEL_VIOLATED = "MINIMUM_TRAVEL_VIOLATED"


ABORT_REASON_PRECEDENCE: Final[tuple[AbortReason, ...]] = (
    AbortReason.KEYBOARD_BLOCKED,
    AbortReason.UNSAFE_ANCHOR,
    AbortReason.OUT_OF_BOUNDS,
    AbortReason.MINIMUM_TRAVEL_VIOLATED,
    AbortReason.NO_VISUAL_CHANGE,
    AbortReason.CAPTURE_FAILED,
    AbortReason.DEVICE_FAILED,
)


DEFAULT_SWIPE_RETRY_ENABLED: Final[bool] = True
DEFAULT_SWIPE_RETRY_DIRECTION: Final[RetryDirection] = RetryDirection.INWARD
DEFAULT_SWIPE_RETRY_MAGNITUDES: Final[tuple[float, ...]] = (0.10, 0.20, 0.30)
DEFAULT_SWIPE_MINIMUM_TRAVEL: Final[int] = 200
DEFAULT_SWIPE_MINIMUM_TRAVEL_FLOOR: Final[int] = 50
