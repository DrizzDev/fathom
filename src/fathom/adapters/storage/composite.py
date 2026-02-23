from __future__ import annotations

import asyncio
from logging import getLogger
from typing import Any, Dict, List, Optional

from fathom.interfaces.storage import StoragePort

logger = getLogger(__name__)


class CompositeStorage(StoragePort):
    """
    Storage adapter that writes to multiple underlying storage ports concurrently.
    Returns the identifier from the primary (first) storage port.
    """

    def __init__(self, storages: List[StoragePort]) -> None:
        """
        Initialize composite storage adapter.
        """

        self.__storages = storages

    async def save(self, *, data: bytes, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Save artifact to all configured storage ports.
        Returns the identifier from the first storage port.
        """

        if not self.__storages:
            return ""

        async def __save_safe(storage: StoragePort) -> Optional[str]:
            try:
                return await storage.save(data=data, metadata=metadata)
            except Exception as exception:
                logger.exception(
                    f"Failed to save to storage {type(storage).__name__}: {exception}",
                    stack_info=True,
                )
                return None

        results = await asyncio.gather(*[__save_safe(storage) for storage in self.__storages])

        # Return first successful result, or empty string
        for result in results:
            if result:
                return result

        return ""
