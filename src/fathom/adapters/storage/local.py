"""Local storage adapter - wraps existing local storage logic."""

from __future__ import annotations

from typing import Any, Dict, Optional

from fathom.infrastructure.storage.local import LocalImageStorage
from fathom.interfaces.storage import StoragePort


class LocalStorage(StoragePort):
    """
    Local filesystem adapter for storage.

    This adapter wraps the existing LocalImageStorage logic without modifications.
    All code delegates to existing implementation to preserve exact behavior.
    """

    def __init__(self) -> None:
        """Initialize local storage adapter."""
        # Use existing implementation as-is
        self.__storage = LocalImageStorage()

    async def save(self, *, data: bytes, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Save artifact and return identifier.

        Delegates to existing LocalImageStorage implementation.
        """
        return await self.__storage.save(data=data, metadata=metadata)
