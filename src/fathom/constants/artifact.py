from __future__ import annotations

from typing import Final


class ArtifactQueue:
    """
    Background-queue tuning constants for the artifact pipeline.
    """

    CAPACITY: Final[int] = 64
    DRAIN_TIMEOUT_SECONDS: Final[float] = 10.0


class ArtifactDirectory:
    """
    Canonical asset directory names beneath ``assets/``.
    """

    XMLS: Final[str] = "xmls"
    TRACES: Final[str] = "traces"
    HISTORY: Final[str] = "history"
    ANNOTATED: Final[str] = "annotated"
    SCREENSHOT: Final[str] = "screenshot"


class StorageBackend:
    """
    Backend identifiers accepted by :class:`StorageConfiguration.backends`.
    """

    LOCAL: Final[str] = "LOCAL"
    CLOUD: Final[str] = "CLOUD"


class ArtifactFilename:
    """
    Filename grammar shared by every pipeline-written artifact.
    """

    STEP_DIGITS: Final[int] = 3
    SEPARATOR: Final[str] = "__"
    TIMESTAMP_FORMAT: Final[str] = "%Y-%m-%dT%H-%M-%SZ"


class ArtifactComponent:
    """
    Structured-logging ``component`` values used by every artifact pipeline emitter.
    """

    PIPELINE: Final[str] = "artifact.pipeline"
    SINK_EFS: Final[str] = "artifact.sink.efs"
    RENDERER: Final[str] = "artifact.renderer"
    SINK_NOOP: Final[str] = "artifact.sink.noop"
    SINK_CLOUD: Final[str] = "artifact.sink.cloud"


class ArtifactEvent:
    """
    Dotted event names used in structured logs. Keeps the event vocabulary discoverable in one place.
    """

    EMIT_STAGED: Final[str] = "artifact.emit.staged"
    EMIT_REJECTED: Final[str] = "artifact.emit.rejected"

    UPLOAD_FAILED: Final[str] = "artifact.upload.failed"
    UPLOAD_STARTED: Final[str] = "artifact.upload.started"
    UPLOAD_SUCCEEDED: Final[str] = "artifact.upload.succeeded"

    RENDER_FAILED: Final[str] = "artifact.render.failed"
    LOCAL_CLEANUP: Final[str] = "artifact.local.cleanup"

    DRAIN_STARTED: Final[str] = "artifact.drain.started"
    DRAIN_COMPLETED: Final[str] = "artifact.drain.completed"
    DRAIN_TIMED_OUT: Final[str] = "artifact.drain.timed_out"
