"""Cloud storage adapter - wraps existing cloud storage logic."""

from __future__ import annotations

from typing import Any, Dict, Optional

from fathom.interfaces import IImageStorage
from fathom.interfaces.storage import StoragePort


class CloudStorage(StoragePort):
    """
    Cloud storage adapter.

    Wraps an IImageStorage implementation.
    """

    def __init__(self, storage: IImageStorage) -> None:
        """Initialize cloud storage adapter with injected storage infrastructure."""
        self.__storage = storage

    async def save(self, *, data: bytes, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Save artifact to cloud and return identifier (URI).
        """
        return await self.__storage.save(data=data, metadata=metadata)
