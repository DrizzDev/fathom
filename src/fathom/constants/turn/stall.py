from __future__ import annotations

from enum import IntEnum, StrEnum


class StallState(StrEnum):
    """
    Classification of the recent action stream's momentum.
    """

    FLOWING = "FLOWING"
    STALLED = "STALLED"
    UNCERTAIN = "UNCERTAIN"


class StallThreshold(IntEnum):
    """
    Trailing non-progress streak at which the action stream counts as stalled.
    """

    STREAK = 3
