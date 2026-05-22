from __future__ import annotations

from importlib import import_module
from typing import Any, Dict, Tuple


class _ExecutionExports:
    """
    Lazy package exports for execution components.
    """

    __MAP: Dict[str, Tuple[str, str]] = {
        "AdaptiveScrollSupervisor": ("fathom.core.execution.scroll", "AdaptiveScrollSupervisor"),
        "ScrollCommandSupervisor": ("fathom.core.execution.scroll", "ScrollCommandSupervisor"),
        "ExecutionEngine": ("fathom.core.execution.engine", "ExecutionEngine"),
    }

    @classmethod
    def get(cls, *, name: str) -> Any:
        """
        Resolve one execution export lazily.
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

    return _ExecutionExports.get(name=name)


__all__ = ["AdaptiveScrollSupervisor", "ScrollCommandSupervisor", "ExecutionEngine"]
