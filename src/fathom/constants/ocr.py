from __future__ import annotations

from enum import StrEnum


class OcrLevel(StrEnum):
    """
    Document AI layout hierarchy level an :class:`OcrToken` represents. Used so downstream layers can attribute snaps
    to the level of structural merging Document AI surfaced (single word, row-merged phrase, multi-row semantic block).
    """

    LINE = "LINE"
    TOKEN = "TOKEN"  # nosec B105
    PARAGRAPH = "PARAGRAPH"


class OcrConfidence(StrEnum):
    """
    Coarse confidence band assigned to an OCR-detected token.
    """

    LOW = "low"
    HIGH = "high"
    MEDIUM = "medium"
