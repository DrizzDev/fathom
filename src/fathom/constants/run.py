from __future__ import annotations

from enum import StrEnum


class ExecutionMode(StrEnum):
    """
    Supported Fathom execution modes.
    """

    INTENT = "INTENT"
    EXPLORATION = "EXPLORATION"


class SignalAdapterType(StrEnum):
    """
    Host interaction adapter types.
    """

    SOCKET = "socket"
    INTERACTIVE = "interactive"


class TargetKind(StrEnum):
    """
    Supported runtime target kinds.
    """

    DEVICE = "DEVICE"
