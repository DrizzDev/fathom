from __future__ import annotations

from enum import Enum


class ActionType(str, Enum):
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
    SCROLL = "scroll"
    COMPLETE = "complete"
    LONG_PRESS = "long_press"


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

    STUCK = "stuck"
    ERROR = "error"
    TIMEOUT = "timeout"
    CONTINUE = "continue"
    COMPLETE = "complete"
