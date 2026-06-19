from __future__ import annotations

from enum import IntEnum, StrEnum
from typing import Final


class LocalizationGridScale(IntEnum):
    """
    Edge bounds of the normalized integer grid used by the vision localizer.
    """

    MINIMUM = 0
    MAXIMUM = 1000


class RegionalEvidenceDecision(StrEnum):
    """
    Decision taken by ``RegionalEvidenceMatcher`` for one evaluation call.
    """

    RESOLVED = "RESOLVED"
    EMPTY_TARGET = "EMPTY_TARGET"
    RECALL_BELOW_FLOOR = "RECALL_BELOW_FLOOR"
    DENSITY_BELOW_FLOOR = "DENSITY_BELOW_FLOOR"
    NO_GEOMETRIC_SIGNAL = "NO_GEOMETRIC_SIGNAL"
    NO_IN_REGION_CLUSTER = "NO_IN_REGION_CLUSTER"
    FUSED_SCORE_BELOW_FLOOR = "FUSED_SCORE_BELOW_FLOOR"


LAYOUT_MIN_WORD_LENGTH_FOR_FUZZ: Final[int] = 3
LAYOUT_MIN_TOKEN_CONFIDENCE: Final[float] = 0.5
LAYOUT_PHRASE_MATCH_THRESHOLD: Final[float] = 0.8
LAYOUT_PER_WORD_SIMILARITY_THRESHOLD: Final[float] = 0.80

LAYOUT_MAX_HEIGHT_RATIO: Final[float] = 2.5
LAYOUT_MAX_ROW_OFFSET_RATIO: Final[float] = 0.5
LAYOUT_MAX_HORIZONTAL_GAP_RATIO: Final[float] = 2.0
