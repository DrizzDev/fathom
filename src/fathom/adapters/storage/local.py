"""Local storage adapter - wraps existing local storage logic."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional

from fathom.infrastructure.storage.local import LocalImageStorage
from fathom.interfaces.storage import StoragePort

if TYPE_CHECKING:
    from fathom.base.paths import SharedPathManager


class LocalStorage(StoragePort):
    """
    Local filesystem adapter for storage.

    This adapter wraps the existing LocalImageStorage logic without modifications.
    All code delegates to existing implementation to preserve exact behavior.
    """

    def __init__(self, path_manager: SharedPathManager) -> None:
        """Initialize local storage adapter."""
        # Use existing implementation as-is
        self.__storage = LocalImageStorage(path_manager=path_manager)

    async def save(self, *, data: bytes, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Save artifact and return identifier.

        Delegates to existing LocalImageStorage implementation.
        """
        return await self.__storage.save(data=data, metadata=metadata)
