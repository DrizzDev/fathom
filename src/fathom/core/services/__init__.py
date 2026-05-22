from __future__ import annotations

from importlib import import_module
from typing import Any, Dict, Tuple


class _ServiceExports:
    """
    Lazy package exports for core services.
    """

    __MAP: Dict[str, Tuple[str, str]] = {
        "HierarchyService": ("fathom.core.services.hierarchy", "HierarchyService"),
        "HistoryService": ("fathom.core.services.history", "HistoryService"),
        "ToolResponseParser": ("fathom.core.services.parsing", "ToolResponseParser"),
        "ReferenceResolutionService": (
            "fathom.core.services.resolution",
            "ReferenceResolutionService",
        ),
        "VisionService": ("fathom.core.services.vision", "VisionService"),
    }

    @classmethod
    def get(cls, *, name: str) -> Any:
        """
        Resolve one service lazily.
        """

        if name not in cls.__MAP:
            raise AttributeError(name)
        module_name, attribute = cls.__MAP[name]
        module = import_module(module_name)
        return getattr(module, attribute)


def __getattr__(name: str) -> Any:
    """
    Resolve package exports lazily.
    """

    return _ServiceExports.get(name=name)


__all__ = [
    "HierarchyService",
    "HistoryService",
    "ToolResponseParser",
    "ReferenceResolutionService",
    "VisionService",
]
