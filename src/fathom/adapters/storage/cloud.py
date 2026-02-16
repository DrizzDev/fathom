"""Cloud storage adapter - wraps existing cloud storage logic."""

from __future__ import annotations

from typing import Any, Dict, Optional

from fathom.infrastructure.storage.cloud import GCSImageStorage
from fathom.interfaces.storage import StoragePort
from fathom.schemas.configuration import GeminiConfig


class CloudStorage(StoragePort):
    """
    Cloud storage adapter.

    Wraps GCSImageStorage.
    """

    def __init__(self, configuration: GeminiConfig, credentials: Any) -> None:
        """Initialize cloud storage adapter."""
        self.__storage = GCSImageStorage(configuration=configuration, credentials=credentials)

    async def save(self, *, data: bytes, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Save artifact to cloud and return identifier (URI).
        """
        return await self.__storage.save(data=data, metadata=metadata)
