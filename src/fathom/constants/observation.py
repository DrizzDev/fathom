from __future__ import annotations

from enum import StrEnum


class KeyboardVisibility(StrEnum):
    """
    Tri-state visibility for the soft keyboard.
    """

    VISIBLE = "VISIBLE"
    HIDDEN = "HIDDEN"
    UNKNOWN = "UNKNOWN"
