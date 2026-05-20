from __future__ import annotations

import asyncio
import time
from logging import getLogger
from typing import Any, Dict

from fathom.constants.artifact import ArtifactComponent, ArtifactEvent
from fathom.interfaces.artifact import ArtifactSinkPort
from fathom.interfaces.storage import StoragePort
from fathom.schemas.artifact import ArtifactCategory, ArtifactMetadata, ArtifactReceipt

logger = getLogger(__name__)


class CloudSink(ArtifactSinkPort):
    """
    Sink that uploads to the configured :class:`StoragePort` and
    requests local cleanup when the upload succeeds.

    On any upload failure (network, throttling, auth), the sink reports
    ``local_cleanup=False`` so the pipeline leaves the EFS file in
    place. The next process's ``replay()`` will pick it up.
    """

    def __init__(
        self,
        *,
        storage: StoragePort,
        workflow_id: str,
    ) -> None:
        """
        Bind this sink to the cloud :class:`StoragePort` and run context.
        """

        self.__storage = storage
        self.__workflow_id = workflow_id

    async def persist(
        self,
        *,
        metadata: ArtifactMetadata,
        content: bytes,
    ) -> ArtifactReceipt:
        """
        Upload the artifact and report whether local cleanup is safe.

        When the upload fails (exception or empty identifier returned by
        the underlying composite storage) we set ``local_cleanup=False``
        so the pipeline preserves the EFS copy for replay and the
        operator sees the artifact route on the next run.
        """

        started = time.monotonic()

        upload_context = {
            **self.__log_context(),
            "payload.bytes": len(content),
            "session.id": metadata.session_id,
            "step.number": metadata.step_number,
            "artifact.kind": metadata.kind.value,
        }
        logger.info(
            "Cloud upload started",
            extra={**upload_context, "event": ArtifactEvent.UPLOAD_STARTED},
        )

        try:
            identifier = await self.__storage.save(
                data=content,
                metadata=self.__upload_metadata(metadata=metadata),
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception(
                "Cloud upload failed; leaving EFS copy for replay",
                extra={
                    **upload_context,
                    "event": ArtifactEvent.UPLOAD_FAILED,
                    "duration.ms": int((time.monotonic() - started) * 1000),
                },
            )
            return ArtifactReceipt(identifier="", local_cleanup=False)

        if not identifier:
            logger.error(
                "Cloud upload returned empty identifier; storage backend silently dropped artifact",
                extra={
                    **upload_context,
                    "reason": "None",
                    "event": ArtifactEvent.UPLOAD_FAILED,
                    "duration.ms": int((time.monotonic() - started) * 1000),
                },
            )
            return ArtifactReceipt(identifier="", local_cleanup=False)

        logger.info(
            "Cloud upload succeeded; local copy eligible for cleanup",
            extra={
                **upload_context,
                "event": ArtifactEvent.UPLOAD_SUCCEEDED,
                "duration.ms": int((time.monotonic() - started) * 1000),
                "artifact.identifier": identifier,
            },
        )
        return ArtifactReceipt(identifier=identifier, local_cleanup=True)

    @staticmethod
    def __upload_metadata(*, metadata: ArtifactMetadata) -> Dict[str, Any]:
        """
        Build the storage-side metadata envelope for one artifact upload.
        """

        return {
            "category": ArtifactCategory.for_(kind=metadata.kind),
            "session_id": metadata.session_id,
            "package_name": metadata.package_name,
            "step_number": metadata.step_number,
            "created": metadata.created,
        }

    def __log_context(self) -> Dict[str, Any]:
        """
        Shared structured-logging context for this sink.
        """

        return {
            "component": ArtifactComponent.SINK_CLOUD,
            "workflow.id": self.__workflow_id,
        }
