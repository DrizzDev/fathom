from __future__ import annotations

from enum import StrEnum


class ActionType(StrEnum):
    """
    Types of actions that can be executed on a device.
    """

    TAP = "tap"
    TYPE = "type"
    TEXT = "type"
    BACK = "back"
    HOME = "home"
    WAIT = "wait"

    SWIPE = "swipe"
    SWIPE_UP = "swipe_up"
    SWIPE_DOWN = "swipe_down"
    SWIPE_LEFT = "swipe_left"
    SWIPE_RIGHT = "swipe_right"

    SCROLL = "scroll"
    COMPLETE = "complete"
    LONG_PRESS = "long_press"
    SAVE_MEMORY = "save_memory"
    RETRIEVE_MEMORY = "retrieve_memory"

    INFER = "infer"
    UNKNOWN = "unknown"


class WorkflowStatus(StrEnum):
    """
    Status of workflow execution.
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"

    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class StrategyStatus(StrEnum):
    """
    Status of strategy execution.
    """

    STUCK = "stuck"
    ERROR = "error"
    TIMEOUT = "timeout"
    CONTINUE = "continue"
    COMPLETE = "complete"
