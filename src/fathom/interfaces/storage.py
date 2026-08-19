from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class StoragePort(ABC):
    """
    Persists artifact bytes and returns a retrieval identifier.
    """

    @abstractmethod
    async def save(self, *, data: bytes, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Persist the bytes, optionally tagged with metadata, and return a retrieval identifier.
        """

        raise NotImplementedError
