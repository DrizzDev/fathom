"""Adapters for hexagonal architecture."""

from fathom.adapters.device.adb import ADBDevice
from fathom.adapters.knowledge.sqlite import SQLiteKnowledge
from fathom.adapters.llm.gemini import GeminiLLM
from fathom.adapters.memory.sqlite import SQLiteMemory
from fathom.adapters.signal.noop import NoopSignal
from fathom.adapters.storage.local import LocalStorage
from fathom.adapters.telemetry.structlog import StructlogAdapter

__all__ = [
    "ADBDevice",
    "GeminiLLM",
    "SQLiteMemory",
    "SQLiteKnowledge",
    "NoopSignal",
    "LocalStorage",
    "StructlogAdapter",
]
