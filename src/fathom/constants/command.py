from __future__ import annotations

from enum import StrEnum


class CommandScopeKind(StrEnum):
    """
    Generic execution scope classification used by perception and scroll surfaces.
    """

    LIST = "list"
    SHEET = "sheet"
    UNKNOWN = "unknown"
    CAROUSEL = "carousel"
    VIEWPORT = "viewport"
    CONTAINER = "container"


class CommandBindingOutcome(StrEnum):
    """
    Result category of binding a command proposal to a canonical CommandSuccess.
    """

    BOUND = "BOUND"
    REJECTED = "REJECTED"


class CommandRejection(StrEnum):
    """
    Why a command proposal could not be bound to a canonical CommandSuccess.
    """

    QUOTE_NOT_FOUND = "QUOTE_NOT_FOUND"
    QUOTE_AMBIGUOUS = "QUOTE_AMBIGUOUS"
    OPERATION_UNSUPPORTED = "OPERATION_UNSUPPORTED"
    CHANNEL_NOT_ADMITTED = "CHANNEL_NOT_ADMITTED"
