from __future__ import annotations

from enum import Enum


class ActionType(str, Enum):
    """
    Types of actions that can be executed on a device.
    """

    TAP = "tap"
    TYPE = "type"
    SWIPE = "swipe"
    SCROLL = "scroll"
    LONG_PRESS = "long_press"
    BACK = "back"
    HOME = "home"
    WAIT = "wait"
    COMPLETE = "complete"


class FlowType(str, Enum):
    """
    Type of execution flow.
    """

    INTENT = "intent"
    EXPLORATION = "exploration"


class WorkflowStatus(str, Enum):
    """
    Status of workflow execution.
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class StrategyStatus(str, Enum):
    """
    Status of strategy execution.
    """

    CONTINUE = "continue"
    COMPLETE = "complete"
    STUCK = "stuck"
    TIMEOUT = "timeout"
    ERROR = "error"
