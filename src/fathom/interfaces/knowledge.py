"""Knowledge port interface for application knowledge graph."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from fathom.schemas.actions import Action


class KnowledgePort(ABC):
    """Abstract interface for application knowledge graph."""

    @abstractmethod
    async def add_screen(self, *, screen_id: str, metadata: Dict[str, Any]) -> None:
        """Add screen node to graph."""
        raise NotImplementedError

    @abstractmethod
    async def add_transition(self, *, from_screen: str, to_screen: str, action: Action) -> None:
        """Add transition edge between screens."""
        raise NotImplementedError

    @abstractmethod
    async def find_path(self, *, from_screen: str, to_screen: str) -> Optional[List[Action]]:
        """Find action sequence to reach target screen."""
        raise NotImplementedError

    @abstractmethod
    async def get_neighbors(self, *, screen_id: str) -> List[str]:
        """Get screens reachable from given screen."""
        raise NotImplementedError
