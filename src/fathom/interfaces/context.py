"""
Interface for Context Engines.
Defines the contract for versioned memory management strategies.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class ContextEngine(ABC):
    """
    Abstract base class for memory construction strategies.
    Supports atomic operations for trace recording, branching, and context assembly.
    """

    @abstractmethod
    async def record(self, *, observation: str, thought: str, action: Dict[str, Any]) -> None:
        """Appends a fine-grained reasoning record to the active log."""
        raise NotImplementedError

    @abstractmethod
    async def commit(self, *, summary: str) -> None:
        """Semantically compresses the current log into a versioned unit."""
        raise NotImplementedError

    @abstractmethod
    async def branch(self, *, branch_name: str) -> None:
        """Creates an isolated fork of the reasoning state."""
        raise NotImplementedError

    @abstractmethod
    def get_context(self) -> Dict[str, Any]:
        """Retrieves a multi-tier view of the current reasoning state."""
        raise NotImplementedError

    @abstractmethod
    async def hydrate(self, *, data: Dict[str, Any]) -> None:
        """Restores state from a serialized representation."""
        raise NotImplementedError

    @abstractmethod
    def dehydrate(self) -> Dict[str, Any]:
        """Serializes current state for persistence."""
        raise NotImplementedError

    @property
    @abstractmethod
    def active_log(self) -> List[Dict[str, Any]]:
        """Returns the current uncommitted trace entries."""
        raise NotImplementedError
