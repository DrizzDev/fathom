from __future__ import annotations

from enum import StrEnum


class InteractionAction(StrEnum):
    """
    Actions supported by the remote interaction API.
    """

    TAP = "TAP"
    TYPE = "TYPE"
    DRAG = "DRAG"
    SWIPE = "SWIPE"
    SCROLL = "SCROLL"

    BACK = "BACK"
    HOME = "HOME"
    PINCH = "PINCH"

    GET_XML = "GET_XML"
    GET_DIMENSIONS = "GET_DIMENSIONS"
    GET_SCREENSHOT = "GET_SCREENSHOT"
    GET_CURRENT_PACKAGE = "GET_CURRENT_PACKAGE"


class SwipeSpeed(StrEnum):
    """
    Gesture speed for swipe, scroll, and drag actions.
    """

    SLOW = "slow"
    FAST = "fast"
    MEDIUM = "medium"
