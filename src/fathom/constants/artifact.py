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
    Every annotation payload (merged or per-kind) shares the flat ``annotated/`` root;
    the filename grammar ``step-NNN__<kind>__<iso>.png`` already disambiguates them.
    """

    XMLS: Final[str] = "xmls"
    TRACES: Final[str] = "traces"
    HISTORY: Final[str] = "history"
    ANNOTATED: Final[str] = "annotated"
    SCREENSHOT: Final[str] = "screenshot"
    PERCEPTION: Final[str] = "annotated"
    CV_PERCEPTION: Final[str] = "annotated"
    OCR_PERCEPTION: Final[str] = "annotated"
    ICON_PERCEPTION: Final[str] = "annotated"
    VISION_PERCEPTION: Final[str] = "annotated"
    OVERLAY_PERCEPTION: Final[str] = "annotated"


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
