from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class ContextEngine(ABC):
    """
    Port for the versioned reasoning memory that feeds each planner turn.

    Records are appended cheaply during a step and periodically compressed into versioned units,
    so the planner reads a bounded multi-tier view instead of the full raw trace.
    """

    @abstractmethod
    async def record(self, *, observation: str, thought: str, action: Dict[str, Any]) -> None:
        """
        Appends a fine-grained reasoning record to the active log.
        """

        raise NotImplementedError

    @abstractmethod
    async def commit(self, *, summary: str) -> None:
        """
        Semantically compresses the current log into a versioned unit.
        """

        raise NotImplementedError

    @abstractmethod
    async def branch(self, *, branch_name: str) -> None:
        """
        Creates an isolated fork of the reasoning state.
        """

        raise NotImplementedError

    @abstractmethod
    def get_context(self) -> Dict[str, Any]:
        """
        Retrieves a multi-tier view of the current reasoning state.
        """

        raise NotImplementedError

    @abstractmethod
    async def hydrate(self, *, data: Dict[str, Any]) -> None:
        """
        Restores state from a serialized representation.
        """

        raise NotImplementedError

    @abstractmethod
    def dehydrate(self) -> Dict[str, Any]:
        """
        Serializes current state for persistence.
        """

        raise NotImplementedError

    @property
    @abstractmethod
    def active_log(self) -> List[Dict[str, Any]]:
        """
        Returns the current uncommitted trace entries.
        """

        raise NotImplementedError
