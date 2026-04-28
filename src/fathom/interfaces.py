from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from fathom.schemas.actions import Action
from fathom.schemas.results import AnalysisResult
from fathom.schemas.screens import ScreenState


@runtime_checkable
class IMemoryProvider(Protocol):
    """
    Contract for persistent knowledge storage.
    """

    async def get_all_knowledge(self) -> Dict[str, Any]: ...
    async def retrieve_knowledge(self, visual_hash: str) -> Dict[str, Any]: ...
    async def store_experience(self, visual_hash: str, action: Action, success: bool) -> None: ...
    async def store_observation(self, screen: ScreenState, description: Optional[str]) -> None: ...
    async def store_transition(
        self, source_hash: str, action: Action, destination_hash: str
    ) -> None: ...
    async def retrieve_transitions(self, visual_hash: str) -> List[Dict[str, Any]]: ...


@runtime_checkable
class IKnowledgeGraph(Protocol):
    """
    Contract for the persistent, cross-run knowledge graph.
    """

    async def load(self) -> None: ...
    async def add_screen(self, state: ScreenState, description: Optional[str] = None) -> Any: ...
    async def record_transition(
        self, source_hash: str, action: Action, destination_hash: str
    ) -> None: ...
    def get_neighbors(self, visual_hash: str) -> List[Any]: ...
    def get_unexplored_screens(self, max_visits: int = 2) -> List[Any]: ...
    def get_stats(self) -> Dict[str, Any]: ...
    def has_screen(self, visual_hash: str) -> bool: ...
    def resolve_hash(self, visual_hash: str) -> str: ...
    def export_json(self) -> Dict[str, Any]: ...
    def export_dot(self) -> str: ...
    def export_mermaid(self) -> str: ...
    def export_html(self) -> str: ...


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

    async def generate_structured(
        self,
        system_instruction: str,
        user_content: List[Any],
        tools: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]: ...

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
