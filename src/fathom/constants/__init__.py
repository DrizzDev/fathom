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
from fathom.constants.run import ExecutionMode, SignalAdapterType, TargetKind
from fathom.constants.scope import ContextScope

SWIPE_ACTIONS = frozenset({"swipe_up", "swipe_down", "swipe_left", "swipe_right", "scroll"})

EXECUTABLE_ACTION_PREFIXES = (
    "open_app ",
    "tap ",
    "type ",
    "scroll ",
    "swipe ",
    "wait ",
    "press ",
    "long press ",
)

VALIDATE_PREFIX = "validate"


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


# Action types that operate on a specific UI element at a pixel location.
# Only spatial actions warrant label-ID snapping to ground-truth coordinates.
# Non-spatial actions (wait, validate, complete, etc.) carry no meaningful target bounds.
SPATIAL_ACTION_TYPES: frozenset[ActionType] = frozenset(
    {
        ActionType.TAP,
        ActionType.TYPE,
        ActionType.SWIPE,
        ActionType.SCROLL,
        ActionType.SWIPE_UP,
        ActionType.LONG_PRESS,
        ActionType.SWIPE_DOWN,
        ActionType.SWIPE_LEFT,
        ActionType.SWIPE_RIGHT,
    }
)

# Action types that, when planned during a sub-goal check, indicate the agent is
# actively executing a next-phase task — used to infer opener sub-goal completion.
NEXT_PHASE_ACTION_TYPES: frozenset[ActionType] = frozenset(
    {
        ActionType.TAP,
        ActionType.WAIT,
        ActionType.TYPE,
        ActionType.SWIPE,
    }
)

# Action types that count as "an action was executed" for sub-goal completion signalling.
ACTION_EXECUTED_TYPES: frozenset[ActionType] = frozenset(
    {
        ActionType.TAP,
        ActionType.TYPE,
        ActionType.SWIPE,
        ActionType.SCROLL,
        ActionType.COMPLETE,
    }
)


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
    "FlowType",
    "ActionType",
    "SignalType",
    "TargetKind",
    "FathomEvent",
    "ContextScope",
    "ExecutionMode",
    "WorkflowStatus",
    "StrategyStatus",
    "ExecutionPhase",
    "DevicePlatform",
    "SignalAdapterType",
    "VISUAL_HASH_LENGTH",
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_RETRY_DELAY",
    "SWIPE_ACTIONS",
    "EXECUTABLE_ACTION_PREFIXES",
    "VALIDATE_PREFIX",
    "SPATIAL_ACTION_TYPES",
    "DeviceConnectionType",
    "IOSAutomationBackend",
    "ACTION_EXECUTED_TYPES",
    "BOUNDS_SWIPE_DISTANCE",
    "DEFAULT_SWIPE_DISTANCE",
    "DEFAULT_SWIPE_DURATION",
    "DEFAULT_STABILITY_WAIT",
    "NEXT_PHASE_ACTION_TYPES",
    "DEFAULT_SCROLL_DISTANCE",
]
