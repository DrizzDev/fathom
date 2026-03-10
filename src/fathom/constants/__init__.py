from __future__ import annotations

from enum import StrEnum

from fathom.constants.events import FathomEvent

# Re-export execution constants
from fathom.constants.execution import (
    BOUNDS_SWIPE_DISTANCE,
    DEFAULT_MAX_RETRIES,
    DEFAULT_RETRY_DELAY,
    DEFAULT_SCROLL_DISTANCE,
    DEFAULT_STABILITY_WAIT,
    DEFAULT_SWIPE_DISTANCE,
    DEFAULT_SWIPE_DURATION,
    VISUAL_HASH_LENGTH,
    ExecutionPhase,
    SignalType,
)
from fathom.constants.platform import DeviceConnectionType, DevicePlatform, IOSAutomationBackend
from fathom.constants.scope import ContextScope


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
    VALIDATE = "validate"
    LONG_PRESS = "long_press"
    SAVE_MEMORY = "save_memory"
    RETRIEVE_MEMORY = "retrieve_memory"

    INFER = "infer"
    UNKNOWN = "unknown"
    ASK_USER = "ask_user"


class FlowType(StrEnum):
    """
    Type of execution flow.
    """

    INTENT = "intent"
    EXPLORATION = "exploration"


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


__all__ = [
    "ActionType",
    "FlowType",
    "ContextScope",
    "FathomEvent",
    "WorkflowStatus",
    "StrategyStatus",
    "SignalType",
    "ExecutionPhase",
    "VISUAL_HASH_LENGTH",
    "DEFAULT_SWIPE_DISTANCE",
    "DEFAULT_SCROLL_DISTANCE",
    "BOUNDS_SWIPE_DISTANCE",
    "DEFAULT_SWIPE_DURATION",
    "DEFAULT_STABILITY_WAIT",
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_RETRY_DELAY",
    "DevicePlatform",
    "DeviceConnectionType",
    "IOSAutomationBackend",
]
