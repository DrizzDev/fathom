from __future__ import annotations

from enum import StrEnum


class AuthoringKind(StrEnum):
    """
    Supported authoring task categories handled by the single authoring agent.
    """

    RUN = "RUN"
    STEP = "STEP"
    REPAIR = "REPAIR"


class AuthoringMode(StrEnum):
    """
    Scheduling mode for optional authoring work.
    """

    SYNC = "SYNC"
    ASYNC = "ASYNC"
    DISABLED = "DISABLED"


class AuthoringStatus(StrEnum):
    """
    Outcome status returned by an authoring request.
    """

    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
    GENERATED = "GENERATED"


class AuthoringArtifactKind(StrEnum):
    """
    Artifact categories an authoring task can reference without loading bytes.
    """

    TEXT = "TEXT"
    TRACE = "TRACE"
    IMAGE = "IMAGE"
    MANIFEST = "MANIFEST"


class AuthoringArtifactRole(StrEnum):
    """
    Role an artifact plays in an authoring task.
    """

    LOG = "LOG"
    OCR = "OCR"
    TREE = "TREE"
    OTHER = "OTHER"
    AFTER = "AFTER"
    BEFORE = "BEFORE"
    ANNOTATED = "ANNOTATED"


class AuthoringExampleKind(StrEnum):
    """
    Role of an example supplied to an authoring prompt.
    """

    AVOID = "AVOID"
    PREFERRED = "PREFERRED"
