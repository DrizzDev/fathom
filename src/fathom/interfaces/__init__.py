from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from fathom.interfaces.device import DevicePort
from fathom.interfaces.knowledge import KnowledgePort
from fathom.interfaces.llm import LLMPort
from fathom.interfaces.memory import MemoryPort
from fathom.interfaces.signal import SignalPort
from fathom.interfaces.storage import StoragePort
from fathom.interfaces.telemetry import TelemetryPort
from fathom.schemas.actions import Action
from fathom.schemas.results import AnalysisResult
from fathom.schemas.screens import ScreenState


# Legacy Protocol definitions for backward compatibility
@runtime_checkable
class IMemoryProvider(Protocol):
    """
    Contract for persistent knowledge storage.
    """

    async def get_all_knowledge(self) -> Dict[str, Any]: ...
    async def retrieve_knowledge(self, visual_hash: str) -> Dict[str, Any]: ...
    async def store_experience(self, visual_hash: str, action: Action, success: bool) -> None: ...
    async def store_observation(self, screen: ScreenState, description: Optional[str]) -> None: ...


@runtime_checkable
class ILedger(Protocol):
    """
    Contract for session-based key-value state storage.
    """

    async def set(self, key: str, value: str) -> None: ...
    async def get(self, key: str) -> Optional[str]: ...
    async def get_all(self) -> Dict[str, str]: ...


@runtime_checkable
class IVisionProvider(Protocol):
    """
    Contract for VLM model interactions.
    """

    async def analyze(
        self,
        system_instruction: str,
        user_content: List[Any],
        tools: Optional[Dict[str, Any]] = None,
    ) -> AnalysisResult: ...

    async def cleanup(self) -> None: ...


@runtime_checkable
class IImageStorage(Protocol):
    """
    Contract for asset persistence.
    """

    async def save(self, data: bytes, metadata: Optional[Dict[str, Any]] = None) -> str: ...


@runtime_checkable
class IResponseParser(Protocol):
    """
    Contract for parsing LLM outputs.
    """

    def parse(self, response: Any) -> AnalysisResult: ...


__all__ = [
    "DevicePort",
    "KnowledgePort",
    "LLMPort",
    "MemoryPort",
    "SignalPort",
    "StoragePort",
    "TelemetryPort",
    "IMemoryProvider",
    "ILedger",
    "IVisionProvider",
    "IImageStorage",
    "IResponseParser",
]
