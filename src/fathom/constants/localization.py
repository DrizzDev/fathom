from __future__ import annotations

from enum import IntEnum
from typing import Final


class LocalizationGridScale(IntEnum):
    """
    Edge bounds of the normalized integer grid used by the vision localizer.
    """

    MINIMUM = 0
    MAXIMUM = 1000


LAYOUT_MIN_WORD_LENGTH_FOR_FUZZ: Final[int] = 3
LAYOUT_MIN_TOKEN_CONFIDENCE: Final[float] = 0.5
LAYOUT_PHRASE_MATCH_THRESHOLD: Final[float] = 0.8
LAYOUT_PER_WORD_SIMILARITY_THRESHOLD: Final[float] = 0.80

LAYOUT_MAX_HEIGHT_RATIO: Final[float] = 2.5
LAYOUT_MAX_ROW_OFFSET_RATIO: Final[float] = 0.5
LAYOUT_MAX_HORIZONTAL_GAP_RATIO: Final[float] = 2.0
