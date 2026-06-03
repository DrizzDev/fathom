from __future__ import annotations

from enum import IntEnum


class LocalizationGridScale(IntEnum):
    """
    Edge bounds of the normalized integer grid used by the vision localizer.
    """

    MINIMUM = 0
    MAXIMUM = 1000
