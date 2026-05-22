from __future__ import annotations

from enum import IntEnum, StrEnum
from typing import Final, FrozenSet


class ScrollDirection(StrEnum):
    """
    Supported scroll directions.
    """

    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"


class ScrollVerdictKind(StrEnum):
    """
    Typed result of one observed scroll attempt.
    """

    PROGRESSED = "progressed"
    NO_PROGRESS = "no_progress"
    WRONG_AXIS = "wrong_axis"
    AMBIGUOUS = "ambiguous"


class ScrollStage(IntEnum):
    """
    Ordered stages within one adaptive scroll run.
    """

    CURRENT = 0
    SHIFT = 1
    SHORT = 2


class ScrollEvidenceSource(StrEnum):
    """
    Source that contributed to a scroll verdict or surface hint.
    """

    CORRELATION = "correlation"
    VERIFIER = "verifier"
    SURFACE = "surface"


class SurfaceKind(StrEnum):
    """
    Surface role that influences adaptive scroll planning.
    """

    KEYBOARD = "keyboard"
    OVERLAY = "overlay"
    PROMO = "promo"
    FOOTER = "footer"


DEFAULT_SCROLL_MAX_ATTEMPTS: Final[int] = 3
DEFAULT_SCROLL_HIGH_CONFIDENCE: Final[float] = 0.92
DEFAULT_SCROLL_LOW_CONFIDENCE: Final[float] = 0.20
DEFAULT_SCROLL_MINIMUM_TRANSLATION: Final[int] = 48
DEFAULT_SCROLL_MAXIMUM_TRANSLATION_RATIO: Final[float] = 0.90
DEFAULT_SCROLL_CORRELATION_STEP: Final[int] = 24
DEFAULT_SCROLL_SHIFT_DISTANCE: Final[int] = 180
DEFAULT_SCROLL_SHORT_RATIO: Final[float] = 0.72
DEFAULT_SCROLL_UNSAFE_MARGIN: Final[int] = 24
DEFAULT_SCROLL_MINIMUM_DISTANCE: Final[int] = 260
DEFAULT_SCROLL_SHIFT_PRIMARY_RATIO: Final[float] = 0.05
DEFAULT_SCROLL_SHIFT_SECONDARY_RATIO: Final[float] = 0.10
DEFAULT_SCROLL_SAFE_TOP_RATIO: Final[float] = 0.15
DEFAULT_SCROLL_ATTEMPT_BUDGET: Final[int] = 200000
DEFAULT_SCROLL_LANE_CLEARANCE_X: Final[int] = 84
DEFAULT_SCROLL_LANE_CLEARANCE_Y: Final[int] = 84
DEFAULT_SCROLL_LANE_CANDIDATE_RATIOS: Final[tuple[float, ...]] = (
    0.5,
    0.35,
    0.65,
    0.2,
    0.8,
    0.1,
    0.9,
)
DEFAULT_SCROLL_LANE_TRAVEL_WEIGHT: Final[int] = 8
DEFAULT_SCROLL_LANE_OFFSET_WEIGHT: Final[int] = 2
DEFAULT_SCROLL_LANE_MINIMUM_TRAVEL_RATIO: Final[float] = 0.70
DEFAULT_SCROLL_PROMO_MINIMUM_WIDTH_RATIO: Final[float] = 0.55
DEFAULT_SCROLL_PROMO_MAXIMUM_HEIGHT_RATIO: Final[float] = 0.18
DEFAULT_SCROLL_BOTTOM_BAND_TOP_RATIO: Final[float] = 0.72
DEFAULT_SCROLL_CENTER_WEIGHT: Final[int] = 0
DEFAULT_SCROLL_SIDE_WEIGHT: Final[int] = 1

PROMO_KEYWORDS: Final[FrozenSet[str]] = frozenset(
    {
        "off",
        "offer",
        "apply",
        "save",
        "free",
        "extra",
        "coupon",
        "delivery",
    }
)
