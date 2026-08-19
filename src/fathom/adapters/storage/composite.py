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
        Bind the ordered list of storage ports to fan writes out to.
        """

        self.__storages = storages

    async def save(self, *, data: bytes, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Save artifact to every configured storage port.

        Returns the identifier from the first backend that succeeded.
        Returns ``""`` only when every backend failed; that case is
        explicitly logged so operators can detect silent artifact loss
        (the deployment bug where a configured cloud backend dropped
        uploads without surfacing on dashboards).
        """

        backend_names = [type(storage).__name__ for storage in self.__storages]

        if not self.__storages:
            logger.warning(
                "Composite storage has no backends configured; dropping write",
                extra={
                    "payload.bytes": len(data),
                    "component": "adapters.storage.composite",
                    "event": "storage.composite.save.no_backends",
                    "metadata.category": (metadata or {}).get("category"),
                },
            )
            return ""

        results = await asyncio.gather(
            *(
                self.__save_one(storage=storage, data=data, metadata=metadata)
                for storage in self.__storages
            ),
        )

        succeeded = [name for name, result in zip(backend_names, results, strict=True) if result]
        failed = [name for name, result in zip(backend_names, results, strict=True) if not result]
        primary_identifier = next((result for result in results if result), "")

        log_payload = {
            "payload.bytes": len(data),
            "backends.failed": failed,
            "backends.succeeded": succeeded,
            "identifier": primary_identifier,
            "backends.configured": backend_names,
            "component": "adapters.storage.composite",
            "metadata.category": (metadata or {}).get("category"),
            "metadata.session_id": (metadata or {}).get("session_id"),
            "metadata.step_number": (metadata or {}).get("step_number"),
        }

        if not succeeded:
            logger.error(
                "Composite storage save failed on every backend; artifact lost",
                extra={**log_payload, "event": "storage.composite.save.all_failed"},
            )
        elif failed:
            logger.warning(
                "Composite storage save partially succeeded",
                extra={**log_payload, "event": "storage.composite.save.partial"},
            )
        else:
            logger.info(
                "Composite storage save completed",
                extra={**log_payload, "event": "storage.composite.save.completed"},
            )

        return primary_identifier

    @staticmethod
    async def __save_one(
        *,
        data: bytes,
        storage: StoragePort,
        metadata: Optional[Dict[str, Any]],
    ) -> Optional[str]:
        """
        Invoke one backend, suppressing exceptions while preserving the traceback.
        """

        backend = type(storage).__name__

        try:
            return await storage.save(data=data, metadata=metadata)
        except Exception:
            logger.exception(
                "Composite storage backend raised",
                extra={
                    "backend": backend,
                    "payload.bytes": len(data),
                    "component": "adapters.storage.composite",
                    "event": "storage.composite.save.backend_failed",
                    "metadata.category": (metadata or {}).get("category"),
                },
            )
            return None
