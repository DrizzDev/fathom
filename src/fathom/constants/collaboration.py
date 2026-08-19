from __future__ import annotations

from enum import IntEnum, StrEnum
from typing import Final, FrozenSet, Mapping

DEFAULT_AGENT: Final[str] = "fathom"
DEFAULT_TENANT: Final[str] = "default"
DEFAULT_OPERATOR: Final[str] = "operator"

# Canonical agent actor identifiers used by the recorder and CLI principal resolver.
# Other runtimes (host-driven REST/Temporal hosts) may supply their own ids; these defaults
# exist so the CLI never silently invents one.
DEFAULT_AGENT_ID: Final[str] = "agent:fathom"
DEFAULT_PLANNER_AGENT_ID: Final[str] = "agent:planner"
DEFAULT_EXECUTOR_AGENT_ID: Final[str] = "agent:executor"
DEFAULT_VERIFIER_AGENT_ID: Final[str] = "agent:verifier"

SYSTEM_WORKER_ACTOR_ID: Final[str] = "system:worker"
SYSTEM_POLICY_ACTOR_ID: Final[str] = "system:policy"
SYSTEM_CLIENT_ACTOR_ID: Final[str] = "system:client"
SYSTEM_RECOVERY_ACTOR_ID: Final[str] = "system:recovery"
SYSTEM_ARTIFACT_ACTOR_ID: Final[str] = "system:artifact"
SYSTEM_INTERACTION_ACTOR_ID: Final[str] = "system:interaction"


INTERACTION_BUILDER: Final[str] = "interaction@1"


class ActorKind(StrEnum):
    """
    Supported kinds of identities that can speak, act, or produce output.
    """

    HUMAN = "human"
    AGENT = "agent"
    COORDINATOR = "coordinator"

    TEAM = "team"
    TOOL = "tool"
    SYSTEM = "system"


class MembershipRole(StrEnum):
    """
    Supported actor roles inside a thread.
    """

    OWNER = "owner"
    SYSTEM = "system"
    DELEGATE = "delegate"
    OBSERVER = "observer"
    REQUESTER = "requester"
    RESPONDER = "responder"
    COORDINATOR = "coordinator"


class MembershipScope(StrEnum):
    """
    Supported visibility scopes for a thread membership.
    """

    TASK = "task"
    TEAM = "team"
    ACTOR = "actor"
    SYSTEM = "system"
    THREAD = "thread"


class ThreadState(StrEnum):
    """
    Supported lifecycle states for a thread.
    """

    ACTIVE = "active"
    PAUSED = "paused"
    DELETED = "deleted"
    ARCHIVED = "archived"
    COMPLETED = "completed"


class TaskKind(StrEnum):
    """
    Supported categories of work represented by a task.
    """

    TOOL = "tool"
    AGENT = "agent"
    FATHOM = "fathom"
    SCRIPT = "script"
    ANALYSIS = "analysis"
    DELEGATION = "delegation"
    COORDINATION = "coordination"
    CLARIFICATION = "clarification"


class TaskState(StrEnum):
    """
    Supported lifecycle states for a task.
    """

    QUEUED = "queued"
    FAILED = "failed"
    RUNNING = "running"
    BLOCKED = "blocked"
    WAITING = "waiting"
    EXPIRED = "expired"
    DELETED = "deleted"
    SUCCEEDED = "succeeded"
    CANCELLED = "cancelled"


class ExecutionState(StrEnum):
    """
    Supported lifecycle states for one user intent execution.
    """

    FAILED = "failed"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    CANCELLED = "cancelled"


class TaskCode(StrEnum):
    """
    Supported machine-readable terminal reason codes for tasks.
    """

    TIMEOUT = "timeout"
    COMPLETED = "completed"
    WORKER_LOST = "worker_lost"
    UNKNOWN_ERROR = "unknown_error"
    USER_CANCELLED = "user_cancelled"
    POLICY_BLOCKED = "policy_blocked"
    VALIDATION_FAILED = "validation_failed"


class MessageKind(StrEnum):
    """
    Supported categories of user-visible or semantic communication.
    """

    NOTE = "note"
    RESULT = "result"
    NOTICE = "notice"
    ANSWER = "answer"
    REQUEST = "request"
    QUESTION = "question"
    PROGRESS = "progress"
    INSTRUCTION = "instruction"


class Audience(StrEnum):
    """
    Supported target audiences for messages.
    """

    TASK = "task"
    TEAM = "team"
    ACTOR = "actor"
    THREAD = "thread"
    SYSTEM = "system"


class EventKind(StrEnum):
    """
    Supported categories of lifecycle records.
    """

    ACTOR_JOINED = "actor.joined"

    THREAD_CREATED = "thread.created"
    THREAD_DELETED = "thread.deleted"
    THREAD_ARCHIVED = "thread.archived"
    THREAD_UNARCHIVED = "thread.unarchived"

    TASK_FAILED = "task.failed"
    TASK_OPENED = "task.opened"
    TASK_STARTED = "task.started"
    TASK_BLOCKED = "task.blocked"
    TASK_WAITING = "task.waiting"
    TASK_EXPIRED = "task.expired"
    TASK_DELETED = "task.deleted"
    TASK_DELEGATED = "task.delegated"
    TASK_SUCCEEDED = "task.succeeded"
    TASK_CANCELLED = "task.cancelled"

    MESSAGE_RECORDED = "message.recorded"
    CONTENT_SANITIZED = "content.sanitized"
    CONTENT_CLASSIFIED = "content.classified"

    CONTEXT_BUILT = "context.built"
    ARTIFACT_LINKED = "artifact.linked"

    JOB_FAILED = "job.failed"
    JOB_SCHEDULED = "job.scheduled"
    JOB_COMPLETED = "job.completed"
    JOB_ABANDONED = "job.abandoned"
    JOB_RESCHEDULED = "job.rescheduled"

    RECOVERY_LOST = "recovery.lost"
    CLIENT_DISCONNECTED = "client.disconnected"


class EventSource(StrEnum):
    """
    Supported producers of lifecycle records.
    """

    FATHOM = "fathom"
    POLICY = "policy"
    WORKER = "worker"
    CLIENT = "client"
    ARTIFACT = "artifact"
    RECOVERY = "recovery"
    INTERACTION = "interaction"


EVENT_SOURCE_ACTORS: Final[Mapping[EventSource, str]] = {
    EventSource.FATHOM: DEFAULT_AGENT_ID,
    EventSource.CLIENT: SYSTEM_CLIENT_ACTOR_ID,
    EventSource.POLICY: SYSTEM_POLICY_ACTOR_ID,
    EventSource.WORKER: SYSTEM_WORKER_ACTOR_ID,
    EventSource.ARTIFACT: SYSTEM_ARTIFACT_ACTOR_ID,
    EventSource.RECOVERY: SYSTEM_RECOVERY_ACTOR_ID,
    EventSource.INTERACTION: SYSTEM_INTERACTION_ACTOR_ID,
}


class ArtifactKind(StrEnum):
    """
    Supported categories of persisted output references.
    """

    TRACE = "trace"
    SCRIPT = "script"
    REPORT = "report"
    SCREENSHOT = "screenshot"
    TOOL_OUTPUT = "tool_output"
    MODEL_OUTPUT = "model_output"
    CONTEXT_DEBUG = "context_debug"
    STRUCTURED_LOG = "structured_log"


class ArtifactBackend(StrEnum):
    """
    Supported artifact storage backends.
    """

    LOCAL = "local"
    OBJECT = "object"


class ScriptStatus(StrEnum):
    """
    Supported lifecycle states for reusable generated scripts.
    """

    DRAFT = "draft"
    ACTIVE = "active"
    DELETED = "deleted"
    ARCHIVED = "archived"


class ScriptVersionSource(StrEnum):
    """
    Supported sources for immutable script versions.
    """

    EDITED = "edited"
    IMPORTED = "imported"
    GENERATED = "generated"


class ScriptFormat(StrEnum):
    """
    Supported content formats for reusable generated scripts.
    """

    TEXT_PLAIN = "text/plain"


class Label(StrEnum):
    """
    Supported labels for privacy, safety, memory, display, and retention policy.
    """

    PRIVACY_OTP = "privacy.otp"
    PRIVACY_UPI = "privacy.upi"
    PRIVACY_AUTH = "privacy.auth"
    PRIVACY_EMAIL = "privacy.email"
    PRIVACY_ADDRESS = "privacy.address"
    PRIVACY_PAYMENT = "privacy.payment"
    PRIVACY_CREDENTIAL = "privacy.credential"

    DISPLAY_DEBUG = "display.debug"
    DISPLAY_AUDIT = "display.audit"
    DISPLAY_HIDDEN = "display.hidden"

    MEMORY_SKIP = "memory.skip"
    RETENTION_SHORT = "retention.short"
    SAFETY_UNTRUSTED = "safety.untrusted"


PRIVATE_LABELS: Final[FrozenSet[Label]] = frozenset(
    {
        Label.PRIVACY_OTP,
        Label.PRIVACY_UPI,
        Label.PRIVACY_AUTH,
        Label.PRIVACY_EMAIL,
        Label.PRIVACY_ADDRESS,
        Label.PRIVACY_PAYMENT,
        Label.PRIVACY_CREDENTIAL,
    }
)


class JobKind(StrEnum):
    """
    Supported categories of background work.
    """

    DIGEST = "digest"
    MEMORY = "memory"
    CONTEXT = "context"
    SANITIZE = "sanitize"
    ARTIFACT = "artifact"
    RECOVERY = "recovery"
    EXECUTION = "execution"


class JobCode(StrEnum):
    """
    Supported machine-readable terminal reason codes for jobs.
    """

    COMPLETED = "completed"
    CANCELLED = "cancelled"
    WORKER_LOST = "worker_lost"
    UNKNOWN_ERROR = "unknown_error"
    RETRYABLE_ERROR = "retryable_error"
    PERMANENT_ERROR = "permanent_error"


class JobState(StrEnum):
    """
    Supported lifecycle states for background jobs.
    """

    FAILED = "failed"
    PENDING = "pending"
    CLAIMED = "claimed"
    COMPLETED = "completed"
    ABANDONED = "abandoned"


class PolicyScope(StrEnum):
    """
    Supported scopes for governance policy.
    """

    TENANT = "tenant"
    WORKSPACE = "workspace"


class ContextPurpose(StrEnum):
    """
    Supported purposes for assembled context records.
    """

    AUDIT = "audit"
    DIGEST = "digest"
    EXECUTION = "execution"
    DELEGATION = "delegation"
    CONVERSATION = "conversation"


class IdempotencyState(StrEnum):
    """
    Supported states for request idempotency records.
    """

    FAILED = "failed"
    STARTED = "started"
    COMPLETED = "completed"


class Priority(IntEnum):
    """
    Numeric priority levels for future scheduling.
    """

    LOW = 10
    HIGH = 90
    NORMAL = 50


class Severity(IntEnum):
    """
    Numeric severity levels for future policy and event handling.
    """

    INFO = 10
    ERROR = 90
    WARNING = 50


TERMINAL_TASK_STATES: Final[FrozenSet[TaskState]] = frozenset(
    {
        TaskState.FAILED,
        TaskState.EXPIRED,
        TaskState.DELETED,
        TaskState.SUCCEEDED,
        TaskState.CANCELLED,
    }
)

POLICY_SCOPES_REQUIRING_WORKSPACE: Final[FrozenSet[PolicyScope]] = frozenset(
    {
        PolicyScope.WORKSPACE,
    }
)

TERMINAL_JOB_STATES: Final[FrozenSet[JobState]] = frozenset(
    {
        JobState.FAILED,
        JobState.COMPLETED,
        JobState.ABANDONED,
    }
)

TERMINAL_IDEMPOTENCY_STATES: Final[FrozenSet[IdempotencyState]] = frozenset(
    {
        IdempotencyState.FAILED,
        IdempotencyState.COMPLETED,
    }
)
