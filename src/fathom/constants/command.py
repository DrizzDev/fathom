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
