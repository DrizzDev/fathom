from __future__ import annotations

from typing import Any, Dict, Optional

from fathom.base.paths import SharedPathManager
from fathom.infrastructure.storage.local import LocalImageStorage
from fathom.interfaces.storage import StoragePort


class LocalStorage(StoragePort):
    """
    Local filesystem adapter for storage.
    """

    def __init__(self, path_manager: SharedPathManager) -> None:
        """
        Initialize local storage adapter.
        """
        self.__storage = LocalImageStorage(path_manager=path_manager)

    async def save(self, *, data: bytes, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Save artifact and return identifier.
        """

        return await self.__storage.save(data=data, metadata=metadata)
