from __future__ import annotations

from enum import StrEnum


class BlockReason(StrEnum):
    """
    Machine-readable reason recorded in failure memory.
    """

    TARGET_UNRESOLVED = "target_unresolved"
    TARGET_AMBIGUOUS = "target_ambiguous"
    REPEATED_NO_EFFECT = "repeated_no_effect"
    REPEATED_CURRENT_SCREEN_ACTION = "repeated_current_screen_action"
    KEYBOARD_OCCLUDING = "keyboard_occluding"
    NON_SCROLLABLE_SURFACE = "non_scrollable_surface"
    OVERLAY_STILL_PRESENT = "overlay_still_present"
    TASK_BUDGET_EXCEEDED = "task_budget_exceeded"
    UNSAFE_ACTION = "unsafe_action"
