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
    PERCEPTION: Final[str] = "perception"
    SCREENSHOT: Final[str] = "screenshot"
    CV_PERCEPTION: Final[str] = "cv_perception"
    OCR_PERCEPTION: Final[str] = "ocr_perception"
    ICON_PERCEPTION: Final[str] = "icon_perception"
    VISION_PERCEPTION: Final[str] = "vision_perception"
    OVERLAY_PERCEPTION: Final[str] = "overlay_perception"


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
