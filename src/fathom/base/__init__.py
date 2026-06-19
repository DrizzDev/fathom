from __future__ import annotations

from importlib import import_module
from typing import Any, Dict, Tuple


class _BaseExports:
    """
    Lazy package exports for base utilities.
    """

    __MAP: Dict[str, Tuple[str, str]] = {
        "BaseLogger": ("fathom.base.logger", "BaseLogger"),
        "SharedPathManager": ("fathom.base.paths", "SharedPathManager"),
        "time_it": ("fathom.base.timing", "time_it"),
    }

    @classmethod
    def get(cls, *, name: str) -> Any:
        """
        Resolve one base export lazily.
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

    return _BaseExports.get(name=name)


__all__ = ["BaseLogger", "SharedPathManager", "time_it"]
