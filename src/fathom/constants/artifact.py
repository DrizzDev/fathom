from __future__ import annotations

from typing import Final


class ArtifactQueue:
    """
    Background-queue tuning constants for the artifact pipeline.

    Single source of truth for queue sizing; every consumer reads from
    here so no literal ``64`` or ``10.0`` floats anywhere else in the
    codebase.
    """

    CAPACITY: Final[int] = 64
    DRAIN_TIMEOUT_SECONDS: Final[float] = 10.0


class ArtifactDirectory:
    """
    Canonical asset directory names beneath ``assets/``.

    Only the five folders the existing :class:`SharedPathManager` and
    conversation-branch :class:`ArtifactCatalog` already understand
    are used. New artifact kinds reuse them rather than introducing
    parallel trees the catalog cannot discover.
    """

    SCREENSHOT: Final[str] = "screenshot"
    ANNOTATED: Final[str] = "annotated"
    TRACES: Final[str] = "traces"
    XMLS: Final[str] = "xmls"
    HISTORY: Final[str] = "history"


class StorageBackend:
    """
    Backend identifiers accepted by :class:`StorageConfiguration.backends`.

    Centralised so sink-selection code never spells the literal inline
    and stays in lock-step with the ``Literal["LOCAL", "CLOUD"]`` typed
    field that defines the vocabulary.
    """

    LOCAL: Final[str] = "LOCAL"
    CLOUD: Final[str] = "CLOUD"


class ArtifactFilename:
    """
    Filename grammar shared by every pipeline-written artifact.

    Pattern: ``step-NNN__<kind>__<iso-timestamp>.<ext>``
    Zero-padded step number, double-underscore field separators,
    ISO-8601 UTC timestamp (``YYYY-MM-DDTHH-MM-SSZ`` — colons replaced
    with hyphens so the value is filesystem-safe).
    """

    STEP_DIGITS: Final[int] = 3
    SEPARATOR: Final[str] = "__"
    TIMESTAMP_FORMAT: Final[str] = "%Y-%m-%dT%H-%M-%SZ"


class ArtifactComponent:
    """
    Structured-logging ``component`` values used by every artifact
    pipeline emitter. Kept centralised so log filters can pick out
    artifact traffic without grepping multiple modules.
    """

    PIPELINE: Final[str] = "artifact.pipeline"
    SINK_EFS: Final[str] = "artifact.sink.efs"
    SINK_CLOUD: Final[str] = "artifact.sink.cloud"
    SINK_NOOP: Final[str] = "artifact.sink.noop"
    RENDERER: Final[str] = "artifact.renderer"


class ArtifactEvent:
    """
    Dotted event names used in structured logs. Keeps the event
    vocabulary discoverable in one place.
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
