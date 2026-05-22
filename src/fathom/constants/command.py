from __future__ import annotations

from enum import StrEnum


class CommandExecutionMode(StrEnum):
    """
    Enforcement mode for command-family execution.
    """

    STRICT = "strict"
    FLEXIBLE = "flexible"


class CommandScopeKind(StrEnum):
    """
    Generic execution scope classification.
    """

    LIST = "list"
    SHEET = "sheet"
    UNKNOWN = "unknown"
    CAROUSEL = "carousel"
    VIEWPORT = "viewport"
    CONTAINER = "container"
