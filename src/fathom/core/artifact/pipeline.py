from __future__ import annotations

import asyncio
import contextlib
from datetime import datetime, timezone
from logging import getLogger
from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional, Set, cast

if TYPE_CHECKING:
    from pathlib import Path

from fathom.base.paths import SharedPathManager
from fathom.constants.artifact import ArtifactComponent, ArtifactEvent, ArtifactFilename
from fathom.interfaces.artifact import ArtifactRendererPort, ArtifactSinkPort
from fathom.schemas.artifact import (
    ArtifactKind,
    ArtifactMetadata,
    ArtifactRecord,
    PipelineConfig,
    TracePayload,
)

logger = getLogger(__name__)


class ArtifactPipeline:
    """
    Centralized observer for every artifact-producing lifecycle stage.

    Producers (perception, action, hierarchy, verification, script) call ``emit(record=...)``
    and the pipeline takes ownership of: staging bytes onto the EFS-backed :class:`SharedPathManager`
    synchronously (the durability boundary), dispatching the metadata + bytes to the
    :class:`ArtifactSinkPort` in a background task, and draining pending work at shutdown.
    """

    def __init__(
        self,
        *,
        config: PipelineConfig,
        renderers: Mapping[ArtifactKind, ArtifactRendererPort],
        sink: ArtifactSinkPort,
        path_manager: SharedPathManager,
        workflow_id: str,
    ) -> None:
        """
        Bind the pipeline to its renderers, sink, path manager, and run context.
        """

        self.__config = config
        self.__renderers = renderers
        self.__sink = sink
        self.__path_manager = path_manager
        self.__workflow_id = workflow_id
        self.__pending: Set[asyncio.Task[None]] = set()
        self.__semaphore = asyncio.Semaphore(config.queue.capacity)

    async def emit(self, *, record: ArtifactRecord) -> Optional[Path]:
        """
        Stage durably and enqueue background upload; never raises.

        Returns the EFS payload path on success so producers can wire
        the same path into downstream metadata (``storage_id``,
        ``capture.metadata["path"]``) without performing a second write
        against :class:`StoragePort`. Returns ``None`` when the record
        was rejected or the renderer raised.
        """

        if (renderer := self.__renderers.get(record.payload.kind)) is None:
            logger.warning(
                "No renderer registered for artifact kind; dropping record",
                extra={
                    **self.__log_context(),
                    "event": ArtifactEvent.EMIT_REJECTED,
                    "artifact.kind": record.payload.kind.value,
                },
            )
            return None

        try:
            content = renderer.render(record=record)
        except asyncio.CancelledError:
            raise
        except Exception as exception:
            logger.warning(
                "Artifact renderer raised; dropping record",
                extra={
                    **self.__log_context(),
                    "event": ArtifactEvent.RENDER_FAILED,
                    "artifact.kind": record.payload.kind.value,
                    "error.message": str(exception),
                },
            )
            return None

        metadata = record.metadata()
        payload_path = await asyncio.to_thread(self.__stage_to_efs, record=record, content=content)
        logger.debug(
            "Artifact staged on EFS",
            extra={
                **self.__log_context(),
                "event": ArtifactEvent.EMIT_STAGED,
                "artifact.kind": metadata.kind.value,
                "artifact.payload.path": str(payload_path),
            },
        )

        task = asyncio.create_task(
            self.__dispatch(
                metadata=metadata,
                content=content,
                payload_path=payload_path,
            ),
        )
        self.__pending.add(task)
        task.add_done_callback(self.__pending.discard)
        return payload_path

    async def drain(self) -> None:
        """
        Await pending background tasks up to the configured drain timeout.
        """

        if not self.__pending:
            return

        logger.info(
            "Artifact pipeline drain started",
            extra={
                **self.__log_context(),
                "event": ArtifactEvent.DRAIN_STARTED,
                "pending.count": len(self.__pending),
            },
        )
        try:
            await asyncio.wait_for(
                asyncio.gather(*self.__pending, return_exceptions=True),
                timeout=self.__config.queue.drain_timeout,
            )
            logger.info(
                "Artifact pipeline drain completed",
                extra={**self.__log_context(), "event": ArtifactEvent.DRAIN_COMPLETED},
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Artifact pipeline drain timed out; surviving tasks abandoned",
                extra={
                    **self.__log_context(),
                    "event": ArtifactEvent.DRAIN_TIMED_OUT,
                    "pending.count": len(self.__pending),
                },
            )

    def __stage_to_efs(
        self,
        *,
        record: ArtifactRecord,
        content: bytes,
    ) -> Path:
        """
        Synchronously write the payload bytes to EFS and return the path.
        """

        metadata = record.metadata()
        filename = self.__filename(record=record, metadata=metadata)
        payload_path = self.__path_manager.get_artifact_path(
            kind=metadata.kind,
            session_id=metadata.session_id,
            package_name=metadata.package_name,
            filename=filename,
        )
        payload_path.write_bytes(content)
        return payload_path

    async def __dispatch(
        self,
        *,
        metadata: ArtifactMetadata,
        content: bytes,
        payload_path: Path,
    ) -> None:
        """
        Background worker: hand off to the sink, then clean up locally if asked.
        """

        async with self.__semaphore:
            try:
                receipt = await self.__sink.persist(metadata=metadata, content=content)
            except asyncio.CancelledError:
                raise
            except Exception as exception:
                logger.warning(
                    "Artifact sink raised; leaving EFS copy in place",
                    extra={
                        **self.__log_context(),
                        "event": ArtifactEvent.UPLOAD_FAILED,
                        "artifact.kind": metadata.kind.value,
                        "error.message": str(exception),
                    },
                )
                return

            if not receipt.local_cleanup:
                return

            await asyncio.to_thread(self.__unlink, payload_path=payload_path)
            logger.debug(
                "Local artifact cleanup completed",
                extra={
                    **self.__log_context(),
                    "event": ArtifactEvent.LOCAL_CLEANUP,
                    "artifact.kind": metadata.kind.value,
                    "artifact.payload.path": str(payload_path),
                },
            )

    @staticmethod
    def __unlink(*, payload_path: Path) -> None:
        """
        Remove the EFS payload after a successful upload.
        """

        with contextlib.suppress(FileNotFoundError):
            payload_path.unlink()

    @staticmethod
    def __filename(*, record: ArtifactRecord, metadata: ArtifactMetadata) -> str:
        """
        Build a stable filename in the canonical artifact grammar.

        Pattern: ``step-NNN__<kind>[__attempt-N]__<iso-timestamp-utc>.<ext>``.
        Zero-padded step ensures directory listings sort by step.
        ISO timestamp with hyphenated time (``T18-33-40Z``) keeps the
        value filesystem-safe across platforms.
        """

        extension = ArtifactPipeline.__extension_for(kind=metadata.kind)
        step = str(metadata.step_number).zfill(ArtifactFilename.STEP_DIGITS)
        timestamp = datetime.fromtimestamp(metadata.created / 1000.0, tz=timezone.utc).strftime(
            ArtifactFilename.TIMESTAMP_FORMAT
        )
        milliseconds = metadata.created % 1000
        separator = ArtifactFilename.SEPARATOR
        attempt_suffix = ""
        if metadata.kind is ArtifactKind.TRACE:
            trace_payload = cast("TracePayload", record.payload)
            if trace_payload.attempt is not None:
                attempt_suffix = f"{separator}attempt-{trace_payload.attempt.index}"
        return (
            f"step-{step}{separator}{metadata.kind.value}{attempt_suffix}{separator}"
            f"{timestamp}-{milliseconds:03d}.{extension}"
        )

    @staticmethod
    def __extension_for(*, kind: ArtifactKind) -> str:
        """
        Resolve the on-disk file extension for one :class:`ArtifactKind`.
        """

        if kind == ArtifactKind.HIERARCHY_XML:
            return "xml"
        if kind == ArtifactKind.SCRIPT:
            return "txt"
        return "png"

    def __log_context(self) -> Dict[str, Any]:
        """
        Shared structured-logging context for pipeline emitters.
        """

        return {
            "component": ArtifactComponent.PIPELINE,
            "workflow.id": self.__workflow_id,
        }

    @property
    def pending_count(self) -> int:
        """
        Number of in-flight background tasks (for test / observability).
        """

        return len(self.__pending)
