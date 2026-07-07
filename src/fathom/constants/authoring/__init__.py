from __future__ import annotations

from enum import StrEnum

AUTHORING_DRAFTS_FILENAME = "authoring.drafts.json"


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
    TRACE = "TRACE"
    OTHER = "OTHER"
    AFTER = "AFTER"
    BEFORE = "BEFORE"
    CONTEXT = "CONTEXT"
    ANNOTATED = "ANNOTATED"


class AuthoringTrust(StrEnum):
    """
    Trust label for authoring evidence channels.
    """

    CLAIM = "CLAIM"
    SCREEN = "SCREEN"


class AuthoringExampleKind(StrEnum):
    """
    Role of an example supplied to an authoring prompt.
    """

    AVOID = "AVOID"
    PREFERRED = "PREFERRED"


class AuthoringLexiconCategory(StrEnum):
    """
    UI terminology category supplied to authoring prompts.
    """

    FIELD = "FIELD"
    CONTROL = "CONTROL"
    CONTENT = "CONTENT"
    FEEDBACK = "FEEDBACK"
    CONTAINER = "CONTAINER"
