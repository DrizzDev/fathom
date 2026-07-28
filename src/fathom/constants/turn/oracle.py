from __future__ import annotations

from enum import Enum


class OracleThreshold(float, Enum):
    """
    Confidence bounds for oracle verdict consumption; below the floor a reading degrades to UNCLEAR.
    """

    CONFIDENCE_FLOOR = 0.7
