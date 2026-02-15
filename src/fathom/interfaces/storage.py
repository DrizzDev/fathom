"""Storage port interface for artifact persistence."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class StoragePort(ABC):
    """Abstract interface for artifact persistence."""

    @abstractmethod
    async def save(
        self, *, data: bytes, metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save artifact and return identifier.

        Args:
            data: Binary data to store
            metadata: Optional metadata for organizing storage

        Returns:
            Storage identifier (path, URL, etc.)
        """
        pass
