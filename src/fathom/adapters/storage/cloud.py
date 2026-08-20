from __future__ import annotations

import time
from logging import getLogger
from typing import Any, Dict, Optional

from fathom.interfaces import IImageStorage
from fathom.interfaces.storage import StoragePort

logger = getLogger(__name__)


class CloudStorage(StoragePort):
    """
    Cloud storage adapter.
    """

    def __init__(self, storage: IImageStorage) -> None:
        """
        Bind the adapter to the injected image-storage backend.
        """

        self.__storage = storage

    async def save(self, *, data: bytes, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Save artifact to cloud and return identifier (URI).
        """

        started = time.monotonic()

        context = {
            "component": "adapters.storage.cloud",
            "payload.bytes": len(data),
            "backend": type(self.__storage).__name__,
            "metadata.category": (metadata or {}).get("category"),
            "metadata.session_id": (metadata or {}).get("session_id"),
            "metadata.step_number": (metadata or {}).get("step_number"),
        }

        logger.info(
            "Cloud storage save started",
            extra={**context, "event": "storage.cloud.save.started"},
        )

        try:
            identifier = await self.__storage.save(data=data, metadata=metadata)
        except Exception:
            logger.exception(
                "Cloud storage save failed",
                extra={
                    **context,
                    "event": "storage.cloud.save.failed",
                    "duration.ms": int((time.monotonic() - started) * 1000),
                },
            )
            raise

        logger.info(
            "Cloud storage save completed",
            extra={
                **context,
                "identifier": identifier,
                "event": "storage.cloud.save.completed",
                "duration.ms": int((time.monotonic() - started) * 1000),
            },
        )
        return identifier
