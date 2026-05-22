from __future__ import annotations

from importlib import import_module
from typing import Any, Dict, Tuple


class _AdapterExports:
    """
    Lazy package exports for adapter entry points.
    """

    __MAP: Dict[str, Tuple[str, str]] = {
        "ADBDevice": ("fathom.adapters.device.local.adb", "ADBDevice"),
        "IOSDevice": ("fathom.adapters.device.local.ios", "IOSDevice"),
        "ADBRemoteDeviceAdapter": ("fathom.adapters.device.remote.adb", "ADBRemoteDeviceAdapter"),
        "IOSRemoteDeviceAdapter": ("fathom.adapters.device.remote.ios", "IOSRemoteDeviceAdapter"),
        "SQLiteKnowledge": ("fathom.adapters.knowledge.sqlite", "SQLiteKnowledge"),
        "CacheService": ("fathom.adapters.llm.cache", "CacheService"),
        "GeminiLLM": ("fathom.adapters.llm.gemini", "GeminiLLM"),
        "SQLiteMemory": ("fathom.adapters.memory.sqlite", "SQLiteMemory"),
        "InteractiveSignal": ("fathom.adapters.signal.interactive", "InteractiveSignal"),
        "NoopSignal": ("fathom.adapters.signal.noop", "NoopSignal"),
        "SocketSignal": ("fathom.adapters.signal.socket", "SocketSignal"),
        "TemporalSignalAdapter": ("fathom.adapters.signal.temporal", "TemporalSignalAdapter"),
        "CloudStorage": ("fathom.adapters.storage.cloud", "CloudStorage"),
        "LocalStorage": ("fathom.adapters.storage.local", "LocalStorage"),
        "StructlogAdapter": ("fathom.adapters.telemetry.structlog", "StructlogAdapter"),
    }

    @classmethod
    def get(cls, *, name: str) -> Any:
        """
        Resolve one exported adapter lazily.
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

    return _AdapterExports.get(name=name)


__all__ = [
    "ADBDevice",
    "IOSDevice",
    "GeminiLLM",
    "SQLiteMemory",
    "SQLiteKnowledge",
    "NoopSignal",
    "LocalStorage",
    "CacheService",
    "StructlogAdapter",
    "ADBRemoteDeviceAdapter",
    "IOSRemoteDeviceAdapter",
    "InteractiveSignal",
    "SocketSignal",
    "CloudStorage",
    "TemporalSignalAdapter",
]
