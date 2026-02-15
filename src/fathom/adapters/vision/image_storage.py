"""
Adapter to make StoragePort compatible with IImageStorage interface.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from fathom.interfaces.storage import StoragePort


class ImageStorageAdapter:
    """
    Adapter that makes StoragePort compatible with IImageStorage.
    
    This allows old agent components to work with the new StoragePort interface.
    """

    def __init__(self, storage: StoragePort) -> None:
        """
        Initialize adapter with storage port.
        
        Args:
            storage: Storage port to wrap
        """
        self.__storage = storage

    async def save(
        self, data: bytes, metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Save data and return storage ID."""
        return await self.__storage.save(data=data, metadata=metadata)
