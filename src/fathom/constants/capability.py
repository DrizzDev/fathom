from __future__ import annotations

from enum import StrEnum
from typing import Final, FrozenSet


class ExecutionChannel(StrEnum):
    """
    How a command reaches the world when executed.
    """

    WAIT = "WAIT"
    DEVICE = "DEVICE"
    MEMORY = "MEMORY"
    CAPTURE = "CAPTURE"
    CONTROL = "CONTROL"
    TERMINAL = "TERMINAL"
    OBSERVATION = "OBSERVATION"


class CompletionMode(StrEnum):
    """
    How a sub-goal whose directive is this command reaches completion.
    """

    TERMINAL = "TERMINAL"
    CLAIM_VERIFIED = "CLAIM_VERIFIED"
    SCREEN_VERIFIED = "SCREEN_VERIFIED"
    CLAIM_OR_TIMEOUT = "CLAIM_OR_TIMEOUT"
    CAPTURE_VERIFIED = "CAPTURE_VERIFIED"
    OUTCOME_VERIFIED = "OUTCOME_VERIFIED"


class RecordMode(StrEnum):
    """
    The persisted event category for a command's step record.
    """

    ACTION = "ACTION"
    CAPTURE = "CAPTURE"
    VALIDATION = "VALIDATION"


class RetryMode(StrEnum):
    """
    Which retry policy wraps a command's execution.
    """

    NONE = "NONE"
    OUTER = "OUTER"
    INTERNAL = "INTERNAL"


class TargetRequirement(StrEnum):
    """
    What kind of on-screen target a command must be grounded to.
    """

    NONE = "NONE"
    REGION = "REGION"
    ELEMENT = "ELEMENT"


class PayloadField(StrEnum):
    """
    A structured payload field a command requires the planner to provide.
    """

    TEXT = "TEXT"
    CAPTURE = "CAPTURE"
    SUBJECT = "SUBJECT"
    WAIT_SUBJECT = "WAIT_SUBJECT"
    SCROLL_TARGET = "SCROLL_TARGET"


NON_INTERACTIVE_CHANNELS: Final[FrozenSet[ExecutionChannel]] = frozenset(
    {
        ExecutionChannel.WAIT,
        ExecutionChannel.MEMORY,
        ExecutionChannel.CAPTURE,
        ExecutionChannel.TERMINAL,
        ExecutionChannel.OBSERVATION,
    }
)
