from __future__ import annotations

from datetime import timedelta
from enum import IntEnum, StrEnum
from typing import Final

RECORDER_BUILDER: Final[str] = "recorder@1"

# Idempotency window applied to recorder-created request rows.
# The recorder owns this constant because it controls the retention contract
# for run-scoped idempotency records before the cleanup sweep reclaims them.
REQUEST_EXPIRY_DAYS: Final[int] = 30


# These are build-time defaults, NOT deployment-tunable's:
TIMELINE_MAX_LIMIT: Final[int] = 500
TIMELINE_DEFAULT_LIMIT: Final[int] = 100

CONVERSATION_LIST_MAX_LIMIT: Final[int] = 200
CONVERSATION_LIST_DEFAULT_LIMIT: Final[int] = 50

MESSAGE_LIST_MAX_LIMIT: Final[int] = 500
MESSAGE_LIST_DEFAULT_LIMIT: Final[int] = 100

ARTIFACT_LIST_MAX_LIMIT: Final[int] = 200
ARTIFACT_LIST_DEFAULT_LIMIT: Final[int] = 50

SCRIPT_LIST_MAX_LIMIT: Final[int] = 200
SCRIPT_LIST_DEFAULT_LIMIT: Final[int] = 50

# Server-side cap on the number of root nodes returned by the task-tree endpoint.
# Children under a root are not counted separately; the guard prevents runaway
# fan-out on conversations with pathologically many root tasks.
TASK_TREE_ROOTS_MAX_LIMIT: Final[int] = 100

# Dedicated caps for the /summary projection — applied only by the summary service path
SUMMARY_SCRIPT_LIMIT: Final[int] = 1_000
SUMMARY_MESSAGE_LIMIT: Final[int] = 10_000

# Text encoding used to measure script content size in bytes for the ScriptView wire payload.
# Scripts are persisted as text and reported with a byte size computed against this canonical encoding.
SCRIPT_CONTENT_ENCODING: Final[str] = "utf-8"

# Filename the runner writes the exported script content under. The artifact
# catalog and runner use this constant to keep the script-export path single-sourced.
SCRIPT_CONTENT_FILENAME: Final[str] = "script.txt"

# Upper bound enforced on user-supplied thread title prefix searches before
# the query reaches the storage adapter; keeps unbounded scans off the floor.
THREAD_TITLE_PREFIX_MAX_LENGTH: Final[int] = 200

# Upper bound enforced on stored thread titles at the schema boundary.
THREAD_TITLE_MAX_LENGTH: Final[int] = 256

# Opaque cursor envelope version. Bumped when the cursor payload format changes
# in a way that older cursors cannot be decoded.
CURSOR_VERSION: Final[str] = "v1"

# Hex length used only to detect raw SHA-256 hashes that should not be rendered as human-readable conversation summaries.
SHA256_HEX_LENGTH: Final[int] = 64


# Default retention windows used by the conversation cleanup service. Hosts override these per-tenant via CleanupRequest fields.
# Stored as timedelta so the unit is encoded in the type rather than the identifier suffix.
CLEANUP_DEFAULT_BATCH_LIMIT: Final[int] = 1_000
CLEANUP_DEFAULT_EVENT_RETENTION: Final[timedelta] = timedelta(days=90)
CLEANUP_DEFAULT_IDEMPOTENCY_RETENTION: Final[timedelta] = timedelta(days=7)
CLEANUP_DEFAULT_TERMINAL_JOB_RETENTION: Final[timedelta] = timedelta(days=14)
CLEANUP_DEFAULT_SOFT_DELETED_RETENTION: Final[timedelta] = timedelta(days=30)


class EntryKind(StrEnum):
    """
    Renderable conversation timeline entry categories.
    """

    EVENT = "event"
    MESSAGE = "message"
    CONTEXT = "context"
    ARTIFACT = "artifact"


class TimelineSource(StrEnum):
    """
    Ledger sources consumed by the composite timeline cursor.
    """

    EVENTS = "events"
    CONTEXTS = "contexts"
    MESSAGES = "messages"
    ARTIFACTS = "artifacts"


class SequenceScope(StrEnum):
    """
    Durable sequence namespaces inside one conversation.
    """

    EVENT = "event"
    MESSAGE = "message"


class Visibility(StrEnum):
    """
    Timeline visibility modes for client, debug, and audit rendering.
    """

    USER = "user"
    DEBUG = "debug"
    AUDIT = "audit"
    HIDDEN = "hidden"


class VisibilityRank(IntEnum):
    """
    Numeric visibility ordering used for deterministic timeline filtering.
    """

    USER = 10
    DEBUG = 20
    AUDIT = 30
    HIDDEN = 40


class ConversationFailureReason(StrEnum):
    """
    Stable client-facing reasons for conversation API failures.
    """

    RUN_NOT_FOUND = "RUN.NOT_FOUND"
    TENANT_REQUIRED = "TENANT_REQUIRED"
    AMBIGUOUS_SCOPE = "AMBIGUOUS_SCOPE"
    THREAD_NOT_FOUND = "THREAD.NOT_FOUND"
    CONVERSATION_REQUIRED = "CONVERSATION_REQUIRED"
    SUMMARY_LIMIT_EXCEEDED = "SUMMARY.LIMIT_EXCEEDED"


class RecorderEvent(StrEnum):
    """
    Stable recorder operation event names.
    """

    RUN_STARTED = "conversation.run.started"
    RUN_FINISHED = "conversation.run.finished"
    RUN_FAILED = "conversation.run.failed"
    STEP_STARTED = "conversation.step.started"
    STEP_FINISHED = "conversation.step.finished"
    SUBTASK_STARTED = "conversation.subtask.started"
    SUBTASK_FINISHED = "conversation.subtask.finished"
    ANALYSIS_RECORDED = "conversation.analysis.recorded"
    HITL_QUESTION = "conversation.hitl.question"
    HITL_ANSWER = "conversation.hitl.answer"
    ARTIFACT_LINKED = "conversation.artifact.linked"
    SCRIPT_SAVED = "conversation.script.saved"
    CONTEXT_BUILT = "conversation.context.built"
    TIMELINE_PROGRESS_RECORDED = "conversation.timeline.progress.recorded"
    TIMELINE_PROGRESS_FAILED = "conversation.timeline.progress.failed"


class RunScriptOutcomeStatus(StrEnum):
    """
    Stable script lookup dispositions returned alongside the script payload.
    """

    AVAILABLE = "AVAILABLE"
    IN_FLIGHT = "IN_FLIGHT"
    NOT_FOUND = "NOT_FOUND"


class RunState(StrEnum):
    """
    Client-facing lifecycle state of one run inside a conversation.
    """

    FAILED = "failed"
    UNKNOWN = "unknown"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    CANCELLED = "cancelled"


class ProgressStatus(StrEnum):
    """
    Client-facing lifecycle state of one progress milestone.
    """

    FAILED = "failed"
    COMPLETED = "completed"
