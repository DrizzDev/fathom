from __future__ import annotations

from datetime import datetime  # noqa: TC003
from enum import StrEnum
from typing import Dict, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from fathom.constants.collaboration import (
    ActorKind,
    ArtifactBackend,
    ArtifactKind,
    Audience,
    ContextPurpose,
    EventKind,
    EventSource,
    IdempotencyState,
    JobCode,
    JobKind,
    JobState,
    Label,
    MembershipRole,
    MembershipScope,
    MessageKind,
    PolicyScope,
    ScriptFormat,
    ScriptStatus,
    ScriptVersionSource,
    TaskCode,
    TaskKind,
    TaskState,
    ThreadState,
)
from fathom.constants.conversation import (
    ARTIFACT_LIST_DEFAULT_LIMIT,
    ARTIFACT_LIST_MAX_LIMIT,
    CLEANUP_DEFAULT_BATCH_LIMIT,
    CONVERSATION_LIST_DEFAULT_LIMIT,
    CONVERSATION_LIST_MAX_LIMIT,
    MESSAGE_LIST_DEFAULT_LIMIT,
    SCRIPT_LIST_DEFAULT_LIMIT,
    SUMMARY_MESSAGE_LIMIT,
    SUMMARY_SCRIPT_LIMIT,
    THREAD_TITLE_PREFIX_MAX_LENGTH,
    TIMELINE_DEFAULT_LIMIT,
    TIMELINE_MAX_LIMIT,
)


class SortOrder(StrEnum):
    """
    Sort direction shared by paginated read queries that expose ordering.
    """

    ASC = "asc"
    DESC = "desc"


class Metadata(BaseModel):
    """
    Optional non-critical JSON metadata for an entity.
    """

    model_config = ConfigDict(
        frozen=True, extra="forbid", protected_namespaces=(), populate_by_name=True
    )

    entries: Dict[str, JsonValue] = Field(
        default_factory=dict,
        description="Optional metadata entries that are not used for policy or querying.",
    )


class Identity(BaseModel):
    """
    Tenant-scoped identity shared by stored entities.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    id: str = Field(description="Stable entity identifier.")
    tenant: str = Field(description="Tenant that owns the entity.")
    workspace: Optional[str] = Field(default=None, description="Optional workspace boundary.")


class Timing(BaseModel):
    """
    Common lifecycle timestamps and elapsed duration.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    created: datetime = Field(
        alias="created_at", description="Timestamp when the entity was created."
    )
    updated: datetime = Field(
        alias="updated_at", description="Timestamp when the entity was last updated."
    )
    started: Optional[datetime] = Field(
        alias="started_at", default=None, description="Timestamp when work started."
    )
    ended: Optional[datetime] = Field(
        alias="ended_at", default=None, description="Timestamp when work ended."
    )
    elapsed: Optional[int] = Field(
        ge=0,
        default=None,
        description="Elapsed duration in milliseconds.",
    )


class Runtime(BaseModel):
    """
    Optional runtime identity for model-backed or service-backed actors.
    """

    model_config = ConfigDict(
        frozen=True, extra="forbid", protected_namespaces=(), populate_by_name=True
    )

    kind: Optional[str] = Field(default=None, description="Runtime category for the actor.")
    provider: Optional[str] = Field(default=None, description="Runtime provider name.")
    model: Optional[str] = Field(default=None, description="Provider-neutral model reference.")


class Assignment(BaseModel):
    """
    Actor references for task creation and assignment.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    creator: Optional[str] = Field(default=None, description="Actor that created the task.")
    assignee: Optional[str] = Field(default=None, description="Actor assigned to the task.")


class Lineage(BaseModel):
    """
    Task tree references for delegation, retry, and continuation.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    parent: Optional[str] = Field(default=None, description="Direct parent task identifier.")
    root: Optional[str] = Field(default=None, description="Root task identifier for the work tree.")
    origin: Optional[str] = Field(default=None, description="Message that caused the task.")


class Plan(BaseModel):
    """
    Structured objective, reference, plan, and progress for a task.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    objective: str = Field(description="Human-readable objective for the task.")
    reference: Optional[str] = Field(default=None, description="Optional target system reference.")
    plan: Metadata = Field(default_factory=Metadata, description="Structured plan details.")
    progress: Metadata = Field(default_factory=Metadata, description="Structured progress details.")


class Terminal(BaseModel):
    """
    Machine and human-readable terminal outcome for a task.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    code: TaskCode = Field(description="Machine-readable terminal reason code.")
    detail: Optional[str] = Field(
        default=None, description="Human-readable terminal reason detail."
    )


class Outcome(BaseModel):
    """
    Machine and human-readable terminal outcome for a background job.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    code: JobCode = Field(description="Machine-readable terminal reason code.")
    detail: Optional[str] = Field(
        default=None, description="Human-readable terminal reason detail."
    )


class Content(BaseModel):
    """
    Message body, labels, and sanitization metadata.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    body: JsonValue = Field(description="Structured message body.")
    labels: Tuple[Label, ...] = Field(
        default_factory=tuple,
        description="Policy labels attached to content.",
    )
    sanitizer: Optional[str] = Field(
        default=None, description="Applied sanitizer profile or rule name."
    )
    sanitized: Optional[datetime] = Field(
        alias="sanitized_at", default=None, description="Timestamp when content was sanitized."
    )


class Thread(BaseModel):
    """
    Durable collaboration timeline for humans, agents, tools, and tasks.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    identity: Identity = Field(description="Tenant-scoped thread identity.")
    title: Optional[str] = Field(default=None, description="Optional user-facing thread title.")
    state: ThreadState = Field(description="Current thread lifecycle state.")
    digest: Optional[str] = Field(default=None, description="Rolling long-context digest.")
    cursor: Optional[int] = Field(
        default=None, ge=0, description="Last sequence included in the digest."
    )
    creator: Optional[str] = Field(default=None, description="Actor that created the thread.")
    timing: Timing = Field(description="Thread lifecycle timestamps.")
    archived: Optional[datetime] = Field(
        alias="archived_at", default=None, description="Timestamp when archived."
    )
    deleted: Optional[datetime] = Field(
        alias="deleted_at", default=None, description="Timestamp when deleted."
    )
    metadata: Metadata = Field(
        default_factory=Metadata, description="Optional non-critical metadata."
    )


class Actor(BaseModel):
    """
    Identity that can speak, act, coordinate, or produce output.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    identity: Identity = Field(description="Tenant-scoped actor identity.")
    kind: ActorKind = Field(description="Actor category.")
    name: str = Field(description="User-facing actor name.")
    external: Optional[str] = Field(default=None, description="Optional external system reference.")
    runtime: Runtime = Field(default_factory=Runtime, description="Optional runtime identity.")
    skills: Metadata = Field(default_factory=Metadata, description="Structured skill metadata.")
    timing: Timing = Field(description="Actor lifecycle timestamps.")
    metadata: Metadata = Field(
        default_factory=Metadata, description="Optional non-critical metadata."
    )


class Membership(BaseModel):
    """
    Actor relationship to a specific thread.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    identity: Identity = Field(description="Tenant-scoped membership identity.")
    thread: str = Field(description="Thread joined by the actor.")
    actor: str = Field(description="Actor that belongs to the thread.")
    role: MembershipRole = Field(description="Actor role inside the thread.")
    scope: MembershipScope = Field(description="Visibility scope for the membership.")
    joined: datetime = Field(alias="joined_at", description="Timestamp when the actor joined.")
    departed_at: Optional[datetime] = Field(
        default=None, description="Timestamp when the actor left."
    )
    metadata: Metadata = Field(
        default_factory=Metadata, description="Optional non-critical metadata."
    )


class Task(BaseModel):
    """
    Durable unit of agent, tool, human, or Fathom work.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    identity: Identity = Field(description="Tenant-scoped task identity.")
    thread: str = Field(description="Thread that owns the task.")
    assignment: Assignment = Field(description="Creator and assignee references.")
    lineage: Lineage = Field(description="Task tree references.")
    kind: TaskKind = Field(description="Category of work represented by the task.")
    state: TaskState = Field(description="Current task lifecycle state.")
    plan: Plan = Field(description="Objective, reference, plan, and progress.")
    terminal: Optional[Terminal] = Field(
        default=None, description="Terminal outcome when finished."
    )
    summary: Optional[str] = Field(default=None, description="Human-readable task result summary.")
    timing: Timing = Field(description="Task lifecycle timestamps.")
    deleted: Optional[datetime] = Field(
        alias="deleted_at", default=None, description="Timestamp when deleted."
    )
    metadata: Metadata = Field(
        default_factory=Metadata, description="Optional non-critical metadata."
    )


class Message(BaseModel):
    """
    User-visible or semantically meaningful communication.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    identity: Identity = Field(description="Tenant-scoped message identity.")
    thread: str = Field(description="Thread that contains the message.")
    task: Optional[str] = Field(default=None, description="Optional task scoped by the message.")
    author: str = Field(description="Actor that authored the message.")
    reply: Optional[str] = Field(default=None, description="Optional parent message.")
    sequence: int = Field(ge=0, description="Stable sequence for ordering messages and events.")
    kind: MessageKind = Field(description="Message category.")
    audience: Audience = Field(description="Intended audience for the message.")
    content: Content = Field(description="Message body and policy labels.")
    created: datetime = Field(
        alias="created_at", description="Timestamp when the message was recorded."
    )
    deleted: Optional[datetime] = Field(
        alias="deleted_at", default=None, description="Timestamp when deleted."
    )
    metadata: Metadata = Field(
        default_factory=Metadata, description="Optional non-critical metadata."
    )


class Event(BaseModel):
    """
    System or product lifecycle fact for a thread.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    identity: Identity = Field(description="Tenant-scoped event identity.")
    thread: str = Field(description="Thread that contains the event.")
    task: Optional[str] = Field(default=None, description="Optional task scoped by the event.")
    actor: Optional[str] = Field(
        default=None, description="Optional actor associated with the event."
    )
    sequence: int = Field(ge=0, description="Stable sequence for event ordering.")
    kind: EventKind = Field(description="Lifecycle event category.")
    source: EventSource = Field(description="System component that produced the event.")
    payload: Metadata = Field(default_factory=Metadata, description="Structured event payload.")
    created: datetime = Field(
        alias="created_at", description="Timestamp when the event was recorded."
    )
    metadata: Metadata = Field(
        default_factory=Metadata, description="Optional non-critical metadata."
    )


class Artifact(BaseModel):
    """
    Durable reference to generated or captured output.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    identity: Identity = Field(description="Tenant-scoped artifact identity.")
    thread: str = Field(description="Thread that owns the artifact.")
    task: Optional[str] = Field(
        default=None, description="Optional task that produced the artifact."
    )
    producer: Optional[str] = Field(
        default=None, description="Optional actor that produced the artifact."
    )
    kind: ArtifactKind = Field(description="Artifact category.")
    uri: str = Field(description="Stable artifact location.")
    backend: ArtifactBackend = Field(description="Storage backend for the artifact.")
    mime: Optional[str] = Field(default=None, description="Optional media type.")
    size: Optional[int] = Field(default=None, ge=0, description="Artifact size in bytes.")
    retention: Optional[str] = Field(default=None, description="Retention class for the artifact.")
    labels: Tuple[Label, ...] = Field(
        default_factory=tuple,
        description="Policy labels attached to the artifact.",
    )
    created: datetime = Field(
        alias="created_at", description="Timestamp when the artifact was linked."
    )
    deleted: Optional[datetime] = Field(
        alias="deleted_at", default=None, description="Timestamp when the artifact was deleted."
    )
    metadata: Metadata = Field(
        default_factory=Metadata, description="Optional non-critical metadata."
    )


class Script(BaseModel):
    """
    Live reusable script produced from a conversation run.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    identity: Identity = Field(description="Tenant-scoped script identity.")
    thread: str = Field(description="Thread that owns the script.")
    task: Optional[str] = Field(default=None, description="Task that produced the script.")
    artifact: Optional[str] = Field(default=None, description="Export artifact for this script.")
    title: Optional[str] = Field(default=None, description="User-facing script title.")
    format: ScriptFormat = Field(
        default=ScriptFormat.TEXT_PLAIN, description="Script content format."
    )
    status: ScriptStatus = Field(description="Current script lifecycle state.")
    content: str = Field(description="Current editable script content.")
    revision: int = Field(ge=1, description="Latest immutable version number.")
    created_by: Optional[str] = Field(default=None, description="Actor that created the script.")
    updated_by: Optional[str] = Field(default=None, description="Actor that last updated it.")
    timing: Timing = Field(description="Script lifecycle timestamps.")
    deleted: Optional[datetime] = Field(
        alias="deleted_at", default=None, description="Timestamp when deleted."
    )
    metadata: Metadata = Field(
        default_factory=Metadata, description="Optional non-critical metadata."
    )


class ScriptVersion(BaseModel):
    """
    Immutable content snapshot for one script version.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    identity: Identity = Field(description="Tenant-scoped script version identity.")
    script: str = Field(description="Script that owns this version.")
    thread: str = Field(description="Thread that owns the script.")
    task: Optional[str] = Field(default=None, description="Task that produced this version.")
    artifact: Optional[str] = Field(default=None, description="Export artifact for this version.")
    version: int = Field(ge=1, description="Monotonic script version number.")
    source: ScriptVersionSource = Field(description="Source of this script version.")
    content: str = Field(description="Immutable script content for this version.")
    checksum: str = Field(description="SHA-256 checksum of the version content.")
    summary: Optional[str] = Field(default=None, description="Human-readable change summary.")
    actor: Optional[str] = Field(default=None, description="Actor that created this version.")
    created: datetime = Field(
        alias="created_at", description="Timestamp when the version was created."
    )
    metadata: Metadata = Field(
        default_factory=Metadata, description="Optional non-critical metadata."
    )


class Governance(BaseModel):
    """
    Tenant or workspace policy rules for retention, labels, sanitizers, memory, and artifacts.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    retention: Metadata = Field(
        default_factory=Metadata, description="Retention rules by record family."
    )
    labels: Metadata = Field(
        default_factory=Metadata, description="Content and artifact label rules."
    )
    sanitizers: Metadata = Field(
        default_factory=Metadata, description="Sanitization rules by label or scope."
    )
    memories: Metadata = Field(default_factory=Metadata, description="Memory projection rules.")
    artifacts: Metadata = Field(
        default_factory=Metadata, description="Artifact storage and lifecycle rules."
    )


class Policy(BaseModel):
    """
    Governance policy for a tenant or workspace.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    identity: Identity = Field(description="Tenant-scoped policy identity.")
    scope: PolicyScope = Field(description="Policy scope.")
    name: str = Field(description="Policy name.")
    region: Optional[str] = Field(default=None, description="Optional data residency region.")
    governance: Governance = Field(description="Structured governance rules.")
    timing: Timing = Field(description="Policy lifecycle timestamps.")
    metadata: Metadata = Field(
        default_factory=Metadata, description="Optional non-critical metadata."
    )


class Job(BaseModel):
    """
    Durable background work item.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    identity: Identity = Field(description="Tenant-scoped job identity.")
    thread: str = Field(description="Thread that owns the job.")
    task: Optional[str] = Field(default=None, description="Optional task scoped by the job.")
    kind: JobKind = Field(description="Background job category.")
    state: JobState = Field(description="Current job lifecycle state.")
    attempts: int = Field(ge=0, description="Number of processing attempts.")
    owner: Optional[str] = Field(default=None, description="Worker that currently owns the job.")
    locked: Optional[datetime] = Field(
        alias="locked_at", default=None, description="Timestamp when the job was claimed."
    )
    available: datetime = Field(
        alias="available_at", description="Timestamp when the job becomes claimable."
    )
    payload: Metadata = Field(default_factory=Metadata, description="Structured job payload.")
    outcome: Optional[Outcome] = Field(default=None, description="Terminal outcome when finished.")
    timing: Timing = Field(description="Job lifecycle timestamps.")
    metadata: Metadata = Field(
        default_factory=Metadata, description="Optional non-critical metadata."
    )


class CreateThread(BaseModel):
    """
    Request to create a thread.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    identity: Identity = Field(description="Identity for the new thread.")
    title: Optional[str] = Field(default=None, description="Optional user-facing thread title.")
    state: ThreadState = Field(default=ThreadState.ACTIVE, description="Initial thread state.")
    creator: Optional[str] = Field(default=None, description="Actor creating the thread.")
    created: datetime = Field(
        alias="created_at", description="Creation timestamp supplied by the application."
    )
    metadata: Metadata = Field(
        default_factory=Metadata, description="Optional non-critical metadata."
    )


class CreateActor(BaseModel):
    """
    Request to create an actor.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    identity: Identity = Field(description="Identity for the new actor.")
    kind: ActorKind = Field(description="Actor category.")
    name: str = Field(description="User-facing actor name.")
    external: Optional[str] = Field(default=None, description="Optional external system reference.")
    runtime: Runtime = Field(default_factory=Runtime, description="Optional runtime identity.")
    skills: Metadata = Field(default_factory=Metadata, description="Structured skill metadata.")
    created: datetime = Field(
        alias="created_at", description="Creation timestamp supplied by the application."
    )
    metadata: Metadata = Field(
        default_factory=Metadata, description="Optional non-critical metadata."
    )


class JoinThread(BaseModel):
    """
    Request to add an actor membership to a thread.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    identity: Identity = Field(description="Identity for the new membership.")
    thread: str = Field(description="Thread joined by the actor.")
    actor: str = Field(description="Actor joining the thread.")
    role: MembershipRole = Field(description="Actor role inside the thread.")
    scope: MembershipScope = Field(default=MembershipScope.THREAD, description="Visibility scope.")
    joined: datetime = Field(
        alias="joined_at", description="Join timestamp supplied by the application."
    )
    metadata: Metadata = Field(
        default_factory=Metadata, description="Optional non-critical metadata."
    )


class OpenTask(BaseModel):
    """
    Request to open a task.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    identity: Identity = Field(description="Identity for the new task.")
    thread: str = Field(description="Thread that owns the task.")
    assignment: Assignment = Field(description="Creator and assignee references.")
    lineage: Lineage = Field(default_factory=Lineage, description="Task tree references.")
    kind: TaskKind = Field(description="Category of work represented by the task.")
    state: TaskState = Field(default=TaskState.QUEUED, description="Initial task state.")
    plan: Plan = Field(description="Objective, reference, plan, and progress.")
    created: datetime = Field(
        alias="created_at", description="Creation timestamp supplied by the application."
    )
    metadata: Metadata = Field(
        default_factory=Metadata, description="Optional non-critical metadata."
    )


class RecordMessage(BaseModel):
    """
    Request to record a message.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    identity: Identity = Field(description="Identity for the new message.")
    thread: str = Field(description="Thread that contains the message.")
    task: Optional[str] = Field(default=None, description="Optional task scoped by the message.")
    author: str = Field(description="Actor that authored the message.")
    reply: Optional[str] = Field(default=None, description="Optional parent message.")
    sequence: Optional[int] = Field(
        default=None,
        ge=1,
        description=(
            "Optional caller-supplied sequence. None means the store allocates "
            "the next per-thread message sequence atomically. An integer must "
            "be >= 1 and is treated as a deterministic caller-owned sequence."
        ),
    )
    kind: MessageKind = Field(description="Message category.")
    audience: Audience = Field(default=Audience.THREAD, description="Intended audience.")
    content: Content = Field(description="Message body and policy labels.")
    created: datetime = Field(
        alias="created_at", description="Creation timestamp supplied by the application."
    )
    metadata: Metadata = Field(
        default_factory=Metadata, description="Optional non-critical metadata."
    )


class Sanitize(BaseModel):
    """
    Request to replace message content with a sanitized version.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    tenant: str = Field(description="Tenant that owns the message.")
    message: str = Field(description="Message whose content should be sanitized.")
    content: Content = Field(description="Sanitized message body and labels.")
    sanitized: datetime = Field(
        alias="sanitized_at", description="Timestamp when sanitization was applied."
    )


class FinishTask(BaseModel):
    """
    Request to finish a task with a terminal state.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    tenant: str = Field(description="Tenant that owns the task.")
    task: str = Field(description="Task to finish.")
    state: TaskState = Field(description="Terminal state to apply.")
    terminal: Terminal = Field(description="Terminal outcome for the task.")
    summary: Optional[str] = Field(default=None, description="Human-readable task result summary.")
    ended: datetime = Field(
        alias="ended_at", description="Finish timestamp supplied by the application."
    )
    elapsed: int = Field(ge=0, description="Elapsed task duration in milliseconds.")


class LinkArtifact(BaseModel):
    """
    Request to link an artifact to a thread and optional task.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    identity: Identity = Field(description="Identity for the new artifact.")
    thread: str = Field(description="Thread that owns the artifact.")
    task: Optional[str] = Field(
        default=None, description="Optional task that produced the artifact."
    )
    producer: Optional[str] = Field(
        default=None, description="Optional actor that produced the artifact."
    )
    kind: ArtifactKind = Field(description="Artifact category.")
    uri: str = Field(description="Stable artifact location.")
    backend: ArtifactBackend = Field(description="Storage backend for the artifact.")
    mime: Optional[str] = Field(default=None, description="Optional media type.")
    size: Optional[int] = Field(default=None, ge=0, description="Artifact size in bytes.")
    retention: Optional[str] = Field(default=None, description="Retention class for the artifact.")
    labels: Tuple[Label, ...] = Field(
        default_factory=tuple,
        description="Policy labels attached to the artifact.",
    )
    created: datetime = Field(
        alias="created_at", description="Creation timestamp supplied by the application."
    )
    metadata: Metadata = Field(
        default_factory=Metadata, description="Optional non-critical metadata."
    )


class SaveScript(BaseModel):
    """
    Request to create or update a reusable script.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    identity: Identity = Field(description="Identity for the script.")
    thread: str = Field(description="Thread that owns the script.")
    task: Optional[str] = Field(default=None, description="Task that produced the script.")
    artifact: Optional[str] = Field(default=None, description="Export artifact for the content.")
    title: Optional[str] = Field(default=None, description="User-facing script title.")
    format: ScriptFormat = Field(
        default=ScriptFormat.TEXT_PLAIN, description="Script content format."
    )
    status: ScriptStatus = Field(default=ScriptStatus.ACTIVE, description="Script state.")
    content: str = Field(description="Editable script content.")
    source: ScriptVersionSource = Field(
        default=ScriptVersionSource.GENERATED,
        description="Source of the version being saved.",
    )
    summary: Optional[str] = Field(default=None, description="Change summary for audit.")
    actor: Optional[str] = Field(default=None, description="Actor saving the script.")
    created: datetime = Field(
        alias="created_at", description="Timestamp supplied by the application."
    )
    metadata: Metadata = Field(
        default_factory=Metadata, description="Optional non-critical metadata."
    )


class SavePolicy(BaseModel):
    """
    Request to save a tenant or workspace policy.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    identity: Identity = Field(description="Identity for the policy.")
    scope: PolicyScope = Field(description="Policy scope.")
    name: str = Field(description="Policy name.")
    region: Optional[str] = Field(default=None, description="Optional data residency region.")
    governance: Governance = Field(
        default_factory=Governance, description="Structured governance rules."
    )
    created: datetime = Field(
        alias="created_at", description="Creation timestamp supplied by the application."
    )
    metadata: Metadata = Field(
        default_factory=Metadata, description="Optional non-critical metadata."
    )


class ScheduleJob(BaseModel):
    """
    Request to schedule background work.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    identity: Identity = Field(description="Identity for the new job.")
    thread: str = Field(description="Thread that owns the job.")
    task: Optional[str] = Field(default=None, description="Optional task scoped by the job.")
    kind: JobKind = Field(description="Background job category.")
    available: datetime = Field(
        alias="available_at", description="Timestamp when the job becomes claimable."
    )
    payload: Metadata = Field(default_factory=Metadata, description="Structured job payload.")
    created: datetime = Field(
        alias="created_at", description="Creation timestamp supplied by the application."
    )
    metadata: Metadata = Field(
        default_factory=Metadata, description="Optional non-critical metadata."
    )


class ClaimJob(BaseModel):
    """
    Request to claim the next available job.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    tenant: str = Field(description="Tenant that owns the job.")
    owner: str = Field(description="Worker claiming the job.")
    claimed: datetime = Field(description="Timestamp when the worker claims the job.")
    kind: Optional[JobKind] = Field(default=None, description="Optional job kind filter.")
    job: Optional[str] = Field(default=None, description="Optional specific job identifier.")


class FinishJob(BaseModel):
    """
    Request to finish a claimed job.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    tenant: str = Field(description="Tenant that owns the job.")
    job: str = Field(description="Job to finish.")
    owner: str = Field(
        description="Worker owner that holds the lease and is finishing the job.",
    )
    state: JobState = Field(description="Terminal state to apply.")
    outcome: Outcome = Field(description="Terminal outcome for the job.")
    finished: datetime = Field(description="Finish timestamp supplied by the worker.")


class RecoverJob(BaseModel):
    """
    Request to release stale claimed jobs.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    tenant: str = Field(description="Tenant that owns stale jobs.")
    before: datetime = Field(description="Claim timestamp before which jobs are stale.")
    available: datetime = Field(
        alias="available_at", description="Timestamp when recovered jobs become claimable."
    )
    kind: Optional[JobKind] = Field(default=None, description="Optional job kind filter.")
    limit: int = Field(
        gt=0,
        default=100,
        le=CLEANUP_DEFAULT_BATCH_LIMIT,
        description="Maximum number of jobs to recover.",
    )


class RescheduleJob(BaseModel):
    """
    Request to release one claimed job for retry.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    tenant: str = Field(description="Tenant that owns the job.")
    job: str = Field(description="Claimed job to reschedule.")
    owner: str = Field(
        description="Worker owner that holds the lease and is rescheduling the job.",
    )
    attempts: int = Field(ge=0, description="Observed attempt count at reschedule time.")
    available: datetime = Field(
        alias="available_at", description="Timestamp when the job becomes claimable again."
    )
    rescheduled: datetime = Field(description="Timestamp when the worker rescheduled the job.")
    detail: Optional[str] = Field(default=None, description="Optional retry reason.")


class ThreadQuery(BaseModel):
    """
    Query for loading one thread.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    tenant: str = Field(description="Tenant that owns the thread.")
    thread: str = Field(description="Thread identifier to load.")


class CleanupRequest(BaseModel):
    """
    Host-issued retention sweep over the interaction store.

    Each retention window is expressed as a `before` timestamp; records
    older than the window are eligible for deletion. Limits cap the rows
    deleted per scope per call so a single sweep on a large database
    cannot run unbounded.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    tenant: Optional[str] = Field(
        default=None,
        description="Optional tenant filter; when None the sweep is global.",
    )
    idempotency_before: Optional[datetime] = Field(
        default=None,
        description="Idempotency rows whose expires_at < this value are deleted.",
    )
    terminal_jobs_before: Optional[datetime] = Field(
        default=None,
        description=(
            "Jobs in completed/failed/abandoned state with updated_at < this "
            "value are deleted along with their rows."
        ),
    )
    events_before: Optional[datetime] = Field(
        default=None,
        description="Lifecycle events with created_at < this value are deleted.",
    )
    soft_deleted_before: Optional[datetime] = Field(
        default=None,
        description=(
            "Soft-deleted entities (deleted_at IS NOT NULL AND deleted_at < this "
            "value) are physically removed."
        ),
    )
    limit: int = Field(
        ge=1,
        le=100_000,
        default=CLEANUP_DEFAULT_BATCH_LIMIT,
        description=(
            "Max rows deleted per scope per call. Capped at 100k so a "
            "direct caller cannot kick off an unbounded sweep."
        ),
    )


class CleanupResult(BaseModel):
    """
    Per-scope deletion counts produced by one CleanupRequest. Retention sweep
    counts (`*_deleted`) reflect threshold-driven removals; soft-delete purge
    counts (`*_purged`) reflect rows removed during thread purge; cascade
    counts (`*_cascade_purged`, plus memberships/contexts/sequences) reflect
    FK-bound thread dependents removed alongside their parent thread.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    idempotency_deleted: int = Field(
        ge=0,
        default=0,
        description="Expired idempotency/request rows removed by the retention sweep.",
    )
    jobs_deleted: int = Field(
        ge=0,
        default=0,
        description="Terminal jobs removed by the retention sweep.",
    )
    events_deleted: int = Field(
        ge=0,
        default=0,
        description="Lifecycle events removed by the retention sweep.",
    )
    threads_purged: int = Field(
        ge=0,
        default=0,
        description="Soft-deleted thread rows physically removed.",
    )
    tasks_purged: int = Field(
        default=0,
        ge=0,
        description="Soft-deleted task rows physically removed.",
    )
    messages_purged: int = Field(
        ge=0,
        default=0,
        description="Soft-deleted message rows physically removed.",
    )
    artifacts_purged: int = Field(
        ge=0,
        default=0,
        description="Soft-deleted artifact rows physically removed.",
    )
    scripts_purged: int = Field(
        ge=0,
        default=0,
        description="Script rows removed while physically purging parent threads.",
    )
    script_versions_purged: int = Field(
        ge=0,
        default=0,
        description="Script version rows removed while purging parent threads.",
    )
    memberships_purged: int = Field(
        ge=0,
        default=0,
        description="Membership rows removed while physically purging parent threads.",
    )
    contexts_purged: int = Field(
        ge=0,
        default=0,
        description="Context rows removed while physically purging parent threads.",
    )
    jobs_cascade_purged: int = Field(
        ge=0,
        default=0,
        description="Job rows removed while physically purging parent threads.",
    )
    events_cascade_purged: int = Field(
        ge=0,
        default=0,
        description="Event rows removed while physically purging parent threads.",
    )
    sequences_purged: int = Field(
        ge=0,
        default=0,
        description="Thread sequence rows removed while physically purging parent threads.",
    )


class CleanupCascadeResult(BaseModel):
    """
    Counts child rows removed while purging a soft-deleted parent thread.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    threads: int = Field(default=0, ge=0, description="Thread rows physically removed.")
    memberships: int = Field(default=0, ge=0, description="Membership rows removed.")
    contexts: int = Field(default=0, ge=0, description="Context rows removed.")
    scripts: int = Field(default=0, ge=0, description="Script rows removed.")
    script_versions: int = Field(default=0, ge=0, description="Script version rows removed.")

    jobs: int = Field(default=0, ge=0, description="Job rows removed.")
    events: int = Field(default=0, ge=0, description="Event rows removed.")
    sequences: int = Field(default=0, ge=0, description="Sequence rows removed.")


class SoftDeletedPurgeOutcome(BaseModel):
    """
    Aggregate counts produced by one soft-deleted purge sweep.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    cascade: CleanupCascadeResult = Field(
        default_factory=CleanupCascadeResult,
        description="Thread cascade counts (parent + FK-bound dependents).",
    )
    tasks: int = Field(default=0, ge=0, description="Soft-deleted task rows physically removed.")
    messages: int = Field(
        default=0, ge=0, description="Soft-deleted message rows physically removed."
    )
    artifacts: int = Field(
        default=0, ge=0, description="Soft-deleted artifact rows physically removed."
    )


class SetThreadTitle(BaseModel):
    """
    Request to set a thread title only when the existing title is null.

    Idempotent: a non-null title is left unchanged. Used by the host to
    auto-fill a conversation's title from the first run's intent.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    tenant: str = Field(description="Tenant that owns the thread.")
    thread: str = Field(description="Thread identifier to update.")
    title: str = Field(min_length=1, description="Title to set when null.")
    updated: datetime = Field(
        alias="updated_at", description="Timestamp of the host-issued update."
    )


class ThreadTransition(BaseModel):
    """
    Request to transition one thread lifecycle state.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    tenant: str = Field(description="Tenant that owns the thread.")
    thread: str = Field(description="Thread identifier to update.")
    state: ThreadState = Field(description="Target thread lifecycle state.")
    actor: Optional[str] = Field(default=None, description="Actor that requested the transition.")
    updated: datetime = Field(
        alias="updated_at", description="Timestamp of the host-issued update."
    )


class ThreadListQuery(BaseModel):
    """
    Cursor-paginated query for tenant-scoped threads.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    tenant: str = Field(description="Tenant that owns the threads.")
    workspace: Optional[str] = Field(default=None, description="Optional workspace filter.")
    state: Optional[ThreadState] = Field(default=None, description="Optional state filter.")
    include_archived: bool = Field(
        default=False,
        description=(
            "Include archived threads in the result set. Deleted threads are never returned."
        ),
    )
    updated_since: Optional[datetime] = Field(
        default=None,
        description="Only include threads updated at or after this timestamp.",
    )
    updated_until: Optional[datetime] = Field(
        default=None,
        description="Only include threads updated before this timestamp.",
    )
    title: Optional[str] = Field(
        default=None,
        max_length=THREAD_TITLE_PREFIX_MAX_LENGTH,
        description="Optional case-insensitive prefix match against thread titles.",
    )
    cursor: Optional[str] = Field(default=None, description="Opaque pagination cursor.")
    count_total: bool = Field(
        default=True,
        description=(
            "Run a COUNT(*) for total match estimate. Set False to skip the "
            "scan when the caller doesn't need an exact total — page.total "
            "will be 0 in that case."
        ),
    )
    limit: int = Field(
        default=CONVERSATION_LIST_DEFAULT_LIMIT,
        gt=0,
        le=CONVERSATION_LIST_MAX_LIMIT,
        description="Maximum threads to return.",
    )


class ThreadPage(BaseModel):
    """
    Cursor-paginated thread results.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    items: Tuple[Thread, ...] = Field(description="Threads in page order.")
    next: Optional[str] = Field(default=None, description="Opaque next-page cursor.")
    total: int = Field(ge=0, description="Total rows matching the query filters.")


class TaskQuery(BaseModel):
    """
    Query for loading tasks scoped to one thread.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    tenant: str = Field(description="Tenant that owns the tasks.")
    thread: str = Field(description="Thread identifier used to scope tasks.")


class TaskOneQuery(BaseModel):
    """
    Query for loading one tenant-scoped task by identifier.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    tenant: str = Field(description="Tenant that owns the task.")
    task: str = Field(description="Task identifier to load.")


class MessageQuery(BaseModel):
    """
    Query for loading messages scoped to one thread and optional task.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    tenant: str = Field(description="Tenant that owns the messages.")
    thread: str = Field(description="Thread identifier used to scope messages.")
    task: Optional[str] = Field(default=None, description="Optional task identifier filter.")


class MessageCursorQuery(BaseModel):
    """
    Cursor-paginated query for messages scoped to one thread.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    tenant: str = Field(description="Tenant that owns the messages.")
    thread: str = Field(description="Thread identifier used to scope messages.")
    task: Optional[str] = Field(default=None, description="Optional task identifier filter.")
    author: Optional[str] = Field(default=None, description="Optional author filter.")
    kinds: Tuple[MessageKind, ...] = Field(
        default_factory=tuple,
        description="Optional message-kind filter.",
    )
    since: Optional[datetime] = Field(
        default=None,
        description="Only include messages created at or after this timestamp.",
    )
    until: Optional[datetime] = Field(
        default=None,
        description="Only include messages created before this timestamp.",
    )
    cursor: Optional[str] = Field(default=None, description="Opaque pagination cursor.")
    count_total: bool = Field(
        default=True,
        description=(
            "Run a COUNT(*) for total match estimate. Set False to skip the "
            "scan when the caller doesn't need an exact total — page.total "
            "will be 0 in that case."
        ),
    )
    limit: int = Field(
        gt=0,
        le=SUMMARY_MESSAGE_LIMIT,
        default=MESSAGE_LIST_DEFAULT_LIMIT,
        description=(
            "Maximum messages to return. Internal callers can request up to "
            "SUMMARY_MESSAGE_LIMIT for projection reads; the public "
            "MessageListQuery still caps paginated list pages at MESSAGE_LIST_MAX_LIMIT."
        ),
    )
    order: SortOrder = Field(
        default=SortOrder.DESC,
        description=(
            "Sort direction by created timestamp (DESC = newest first, "
            "ASC = oldest first). DESC is the default so chat-style clients "
            "render newest at the bottom and scroll up to fetch older. "
            "Cursor payload is direction-agnostic; callers must keep `order` "
            "consistent across pages or pagination will skip/repeat rows."
        ),
    )


class MessagePage(BaseModel):
    """
    Cursor-paginated message results.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    items: Tuple[Message, ...] = Field(description="Messages in page order.")
    next: Optional[str] = Field(default=None, description="Opaque next-page cursor.")
    total: int = Field(ge=0, description="Total rows matching the query filters.")


class EventQuery(BaseModel):
    """
    Query for loading events scoped to one thread.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    tenant: str = Field(description="Tenant that owns the events.")
    thread: str = Field(description="Thread identifier used to scope events.")
    task: Optional[str] = Field(default=None, description="Optional task identifier filter.")


class EventCursorQuery(BaseModel):
    """
    Cursor-paginated query for lifecycle events scoped to one thread.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    tenant: str = Field(description="Tenant that owns the events.")
    thread: str = Field(description="Thread identifier used to scope events.")
    task: Optional[str] = Field(default=None, description="Optional task identifier filter.")
    actor: Optional[str] = Field(default=None, description="Optional actor filter.")
    kinds: Tuple[EventKind, ...] = Field(
        default_factory=tuple,
        description="Optional event-kind filter.",
    )
    since: Optional[datetime] = Field(
        default=None,
        description="Only include events created at or after this timestamp.",
    )
    until: Optional[datetime] = Field(
        default=None,
        description="Only include events created before this timestamp.",
    )
    cursor: Optional[str] = Field(default=None, description="Opaque pagination cursor.")
    count_total: bool = Field(
        default=True,
        description=(
            "Run a COUNT(*) for total match estimate. Set False to skip the "
            "scan when the caller doesn't need an exact total — page.total "
            "will be 0 in that case."
        ),
    )
    limit: int = Field(
        default=TIMELINE_DEFAULT_LIMIT,
        gt=0,
        le=TIMELINE_MAX_LIMIT,
        description="Maximum events to return.",
    )
    order: SortOrder = Field(
        default=SortOrder.DESC,
        description=(
            "Sort direction by created timestamp (DESC = newest first, "
            "ASC = oldest first). Cursor payload is direction-agnostic; callers must keep `order` consistent across pages or pagination will skip/repeat rows."
        ),
    )


class EventPage(BaseModel):
    """
    Cursor-paginated event results.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    items: Tuple[Event, ...] = Field(description="Events in page order.")
    next: Optional[str] = Field(default=None, description="Opaque next-page cursor.")
    total: int = Field(ge=0, description="Total rows matching the query filters.")


class ArtifactQuery(BaseModel):
    """
    Query for loading artifacts scoped to one thread.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    tenant: str = Field(description="Tenant that owns the artifacts.")
    thread: str = Field(description="Thread identifier used to scope artifacts.")
    task: Optional[str] = Field(default=None, description="Optional task identifier filter.")


class ArtifactCursorQuery(BaseModel):
    """
    Cursor-paginated query for artifacts scoped to one thread.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    tenant: str = Field(description="Tenant that owns the artifacts.")
    thread: str = Field(description="Thread identifier used to scope artifacts.")
    task: Optional[str] = Field(default=None, description="Optional task identifier filter.")
    producer: Optional[str] = Field(default=None, description="Optional producer filter.")
    kinds: Tuple[ArtifactKind, ...] = Field(
        default_factory=tuple,
        description="Optional artifact-kind filter.",
    )
    since: Optional[datetime] = Field(
        default=None,
        description="Only include artifacts created at or after this timestamp.",
    )
    until: Optional[datetime] = Field(
        default=None,
        description="Only include artifacts created before this timestamp.",
    )
    cursor: Optional[str] = Field(default=None, description="Opaque pagination cursor.")
    count_total: bool = Field(
        default=True,
        description=(
            "Run a COUNT(*) for total match estimate. Set False to skip the "
            "scan when the caller doesn't need an exact total — page.total "
            "will be 0 in that case."
        ),
    )
    limit: int = Field(
        default=ARTIFACT_LIST_DEFAULT_LIMIT,
        gt=0,
        le=ARTIFACT_LIST_MAX_LIMIT,
        description="Maximum artifacts to return.",
    )
    order: SortOrder = Field(
        default=SortOrder.DESC,
        description=(
            "Sort direction by created timestamp (DESC = newest first, "
            "ASC = oldest first). DESC is the default so chat-style clients "
            "render newest at the bottom. Cursor payload is direction-agnostic; callers must keep `order` consistent across pages or pagination will skip/repeat rows."
        ),
    )


class ArtifactPage(BaseModel):
    """
    Cursor-paginated artifact results.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    items: Tuple[Artifact, ...] = Field(description="Artifacts in page order.")
    next: Optional[str] = Field(default=None, description="Opaque next-page cursor.")
    total: int = Field(ge=0, description="Total rows matching the query filters.")


class ScriptQuery(BaseModel):
    """
    Query for loading scripts by identity or conversation scope.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    tenant: str = Field(description="Tenant that owns the scripts.")
    thread: Optional[str] = Field(default=None, description="Optional thread filter.")
    script: Optional[str] = Field(default=None, description="Optional script identifier.")
    task: Optional[str] = Field(default=None, description="Optional task identifier filter.")
    artifact: Optional[str] = Field(default=None, description="Optional artifact identifier.")
    include_deleted: bool = Field(
        default=False,
        description="Include soft-deleted scripts when true.",
    )


class ScriptVersionQuery(BaseModel):
    """
    Query for loading immutable versions of one script.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    tenant: str = Field(description="Tenant that owns the script versions.")
    script: str = Field(description="Script whose versions should be loaded.")
    version: Optional[int] = Field(default=None, ge=1, description="Optional version number.")


class ScriptListQuery(BaseModel):
    """
    Cursor-paginated query for scripts scoped to one thread.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    tenant: str = Field(description="Tenant that owns the scripts.")
    thread: str = Field(description="Thread identifier used to scope scripts.")
    task: Optional[str] = Field(default=None, description="Optional task identifier filter.")
    since: Optional[datetime] = Field(
        default=None,
        description="Only include scripts updated at or after this timestamp.",
    )
    until: Optional[datetime] = Field(
        default=None,
        description="Only include scripts updated before this timestamp.",
    )
    cursor: Optional[str] = Field(default=None, description="Opaque pagination cursor.")
    count: bool = Field(
        default=True,
        description=(
            "Whether to run COUNT(*) for the total match estimate. When false the "
            "scan is skipped and page.total is reported as 0."
        ),
    )
    limit: int = Field(
        gt=0,
        le=SUMMARY_SCRIPT_LIMIT,
        default=SCRIPT_LIST_DEFAULT_LIMIT,
        description=(
            "Maximum scripts to return. Internal callers can request up to "
            "SUMMARY_SCRIPT_LIMIT for projection reads; the public ScriptsQuery "
            "still caps paginated list pages at SCRIPT_LIST_MAX_LIMIT."
        ),
    )
    order: SortOrder = Field(
        default=SortOrder.DESC,
        description=(
            "Sort direction by updated timestamp (ASC = oldest first, "
            "DESC = newest first). Cursor payload is direction-agnostic; callers must keep `order` consistent across pages or pagination will skip/repeat rows."
        ),
    )
    include_deleted: bool = Field(
        default=False,
        description="Include soft-deleted scripts when true.",
    )


class ScriptPage(BaseModel):
    """
    Cursor-paginated script results.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    items: Tuple[Script, ...] = Field(description="Scripts in page order.")
    next: Optional[str] = Field(default=None, description="Opaque next-page cursor.")
    total: int = Field(ge=0, description="Total rows matching the query filters.")


class SummaryMessagesQuery(BaseModel):
    """
    Bounded one-shot read of every message of selected kinds in one thread for the /summary projection.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    tenant: str = Field(description="Tenant that owns the messages.")
    thread: str = Field(description="Thread identifier used to scope messages.")
    kinds: Tuple[MessageKind, ...] = Field(
        default_factory=tuple,
        description="Message kinds to include. Empty selects every kind.",
    )


class SummaryScriptsQuery(BaseModel):
    """
    Bounded one-shot read of every script in one thread for the /summary projection.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    tenant: str = Field(description="Tenant that owns the scripts.")
    thread: str = Field(description="Thread identifier used to scope scripts.")


class ContextCursorQuery(BaseModel):
    """
    Cursor-paginated query for context records scoped to one thread.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    tenant: str = Field(description="Tenant that owns the contexts.")
    thread: str = Field(description="Thread identifier used to scope contexts.")
    task: Optional[str] = Field(default=None, description="Optional task identifier filter.")
    consumer: Optional[str] = Field(default=None, description="Optional consumer filter.")
    purpose: Optional[ContextPurpose] = Field(
        default=None,
        description="Optional context purpose filter.",
    )
    since: Optional[datetime] = Field(
        default=None,
        description="Only include contexts created at or after this timestamp.",
    )
    until: Optional[datetime] = Field(
        default=None,
        description="Only include contexts created before this timestamp.",
    )
    cursor: Optional[str] = Field(default=None, description="Opaque pagination cursor.")
    count_total: bool = Field(
        default=True,
        description=(
            "Run a COUNT(*) for total match estimate. Set False to skip the "
            "scan when the caller doesn't need an exact total — page.total "
            "will be 0 in that case."
        ),
    )
    limit: int = Field(
        default=TIMELINE_DEFAULT_LIMIT,
        gt=0,
        le=TIMELINE_MAX_LIMIT,
        description="Maximum contexts to return.",
    )
    order: SortOrder = Field(
        default=SortOrder.DESC,
        description=(
            "Sort direction by created timestamp (DESC = newest first, "
            "ASC = oldest first). Cursor payload is direction-agnostic; callers must keep `order` consistent across pages or pagination will skip/repeat rows."
        ),
    )


class ContextPage(BaseModel):
    """
    Cursor-paginated context results.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    items: Tuple[Context, ...] = Field(description="Contexts in page order.")
    next: Optional[str] = Field(default=None, description="Opaque next-page cursor.")
    total: int = Field(ge=0, description="Total rows matching the query filters.")


class PolicyQuery(BaseModel):
    """
    Query for loading one policy by tenant, workspace, and name.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    tenant: str = Field(description="Tenant that owns the policy.")
    workspace: Optional[str] = Field(default=None, description="Optional workspace boundary.")
    name: str = Field(description="Policy name.")


class JobQuery(BaseModel):
    """
    Query for loading jobs scoped to one tenant.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    tenant: str = Field(description="Tenant that owns the jobs.")
    thread: Optional[str] = Field(default=None, description="Optional thread filter.")
    state: Optional[JobState] = Field(default=None, description="Optional job state filter.")
    kind: Optional[JobKind] = Field(default=None, description="Optional job kind filter.")


class MemoryReference(BaseModel):
    """
    Pointer into an external memory system used for context assembly.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    system: str = Field(description="Name of the external memory system.")
    reference: str = Field(description="System-specific reference identifier.")


class References(BaseModel):
    """
    Reference set assembled into one context record.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    messages: Tuple[str, ...] = Field(default_factory=tuple, description="Message identifiers.")
    events: Tuple[str, ...] = Field(default_factory=tuple, description="Event identifiers.")
    artifacts: Tuple[str, ...] = Field(default_factory=tuple, description="Artifact identifiers.")
    memories: Tuple[MemoryReference, ...] = Field(
        default_factory=tuple,
        description="External memory references.",
    )


class Context(BaseModel):
    """
    Reference-based assembly recipe for a model or tool call.
    """

    model_config = ConfigDict(
        frozen=True, extra="forbid", protected_namespaces=(), populate_by_name=True
    )

    identity: Identity = Field(description="Tenant-scoped context identity.")
    thread: str = Field(description="Thread that owns the context.")
    task: Optional[str] = Field(default=None, description="Optional task scoped by the context.")
    consumer: Optional[str] = Field(default=None, description="Actor that consumes the context.")
    purpose: ContextPurpose = Field(description="Purpose of the assembled context.")
    builder: str = Field(description="Builder name and version that produced the recipe.")
    references: References = Field(description="Set of references included in the context.")
    budget: Metadata = Field(
        default_factory=Metadata, description="Budget decisions for the context."
    )
    filters: Metadata = Field(
        default_factory=Metadata, description="Filter decisions for the context."
    )
    hash: Optional[str] = Field(default=None, description="Optional stable hash of the recipe.")
    provider: Optional[str] = Field(default=None, description="Optional model provider name.")
    model: Optional[str] = Field(default=None, description="Optional model reference.")
    created: datetime = Field(
        alias="created_at", description="Timestamp when the context was assembled."
    )
    expires: Optional[datetime] = Field(
        alias="expires_at", default=None, description="Timestamp when the context expires."
    )
    metadata: Metadata = Field(
        default_factory=Metadata, description="Optional non-critical metadata."
    )


class BuildContext(BaseModel):
    """
    Request to assemble and persist one context recipe.
    """

    model_config = ConfigDict(
        frozen=True, extra="forbid", protected_namespaces=(), populate_by_name=True
    )

    identity: Identity = Field(description="Identity for the new context.")
    thread: str = Field(description="Thread that owns the context.")
    task: Optional[str] = Field(default=None, description="Optional task scoped by the context.")
    consumer: Optional[str] = Field(default=None, description="Actor that consumes the context.")
    purpose: ContextPurpose = Field(description="Purpose of the assembled context.")
    builder: str = Field(description="Builder name and version that produced the recipe.")
    references: References = Field(description="Set of references included in the context.")
    budget: Metadata = Field(
        default_factory=Metadata, description="Budget decisions for the context."
    )
    filters: Metadata = Field(
        default_factory=Metadata, description="Filter decisions for the context."
    )
    hash: Optional[str] = Field(default=None, description="Optional stable hash of the recipe.")
    provider: Optional[str] = Field(default=None, description="Optional model provider name.")
    model: Optional[str] = Field(default=None, description="Optional model reference.")
    created: datetime = Field(
        alias="created_at", description="Creation timestamp supplied by the application."
    )
    expires: Optional[datetime] = Field(
        alias="expires_at", default=None, description="Optional expiry timestamp."
    )
    metadata: Metadata = Field(
        default_factory=Metadata, description="Optional non-critical metadata."
    )


class ContextQuery(BaseModel):
    """
    Query for loading contexts scoped to one thread.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    tenant: str = Field(description="Tenant that owns the contexts.")
    thread: str = Field(description="Thread identifier used to scope contexts.")
    task: Optional[str] = Field(default=None, description="Optional task identifier filter.")
    purpose: Optional[ContextPurpose] = Field(default=None, description="Optional purpose filter.")


class Idempotency(BaseModel):
    """
    Retry-safety record for a tenant-scoped create or update request.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    tenant: str = Field(description="Tenant that owns the idempotency record.")
    key: str = Field(description="Caller-supplied idempotency key.")
    hash: str = Field(description="Stable hash of the original request payload.")
    state: IdempotencyState = Field(description="Current request lifecycle state.")
    response: Optional[JsonValue] = Field(default=None, description="Cached response for replay.")
    created: datetime = Field(
        alias="created_at", description="Timestamp when the record was created."
    )
    expires: datetime = Field(
        alias="expires_at", description="Timestamp when the record may be discarded."
    )
    metadata: Metadata = Field(
        default_factory=Metadata, description="Optional non-critical metadata."
    )


class BeginRequest(BaseModel):
    """
    Request to start an idempotent operation.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    tenant: str = Field(description="Tenant that owns the idempotency record.")
    key: str = Field(description="Caller-supplied idempotency key.")
    hash: str = Field(description="Stable hash of the request payload.")
    created: datetime = Field(
        alias="created_at", description="Creation timestamp supplied by the application."
    )
    expires: datetime = Field(
        alias="expires_at", description="Timestamp when the record may be discarded."
    )
    metadata: Metadata = Field(
        default_factory=Metadata, description="Optional non-critical metadata."
    )

    @model_validator(mode="after")
    def __validate_expiry(self) -> "BeginRequest":
        """
        Require idempotency expiry to be after creation.
        """

        if self.expires <= self.created:
            raise ValueError("Idempotency expires_at must be after created_at.")

        return self


class FinishRequest(BaseModel):
    """
    Request to record the terminal state of an idempotent operation.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    tenant: str = Field(description="Tenant that owns the idempotency record.")
    key: str = Field(description="Caller-supplied idempotency key.")
    state: IdempotencyState = Field(description="Terminal idempotency state.")
    response: Optional[JsonValue] = Field(default=None, description="Cached response for replay.")
    finished: datetime = Field(description="Finish timestamp supplied by the application.")


class IdempotencyQuery(BaseModel):
    """
    Query for loading one tenant-scoped idempotency record.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    tenant: str = Field(description="Tenant that owns the idempotency record.")
    key: str = Field(description="Caller-supplied idempotency key.")


class RunStart(BaseModel):
    """
    Request to begin recording one execution run.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    tenant: str = Field(description="Tenant that owns the run.")
    workspace: Optional[str] = Field(default=None, description="Optional workspace boundary.")
    thread: str = Field(description="Thread used to group related run records.")
    workflow: str = Field(description="Execution workflow identifier.")
    intent: str = Field(description="User goal for the run.")
    package: Optional[str] = Field(default=None, description="Optional target package reference.")
    operator: str = Field(description="Human actor identifier.")
    agent: str = Field(description="Agent actor identifier.")
    started: datetime = Field(alias="started_at", description="Timestamp when recording starts.")
    metadata: Metadata = Field(
        default_factory=Metadata, description="Optional non-critical metadata."
    )


class RunHandle(BaseModel):
    """
    Stable identifiers created for one recorded run.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    tenant: str = Field(description="Tenant that owns the run.")
    workspace: Optional[str] = Field(default=None, description="Optional workspace boundary.")
    thread: str = Field(description="Thread that owns the run.")
    task: str = Field(description="Root task representing the run.")
    workflow: str = Field(description="Execution workflow identifier.")
    operator: str = Field(description="Human actor identifier.")
    agent: str = Field(description="Agent actor identifier.")
    request: str = Field(description="Message identifier for the original request.")


class RunFinish(BaseModel):
    """
    Request to finish recording one execution run.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    handle: RunHandle = Field(description="Stable identifiers for the run.")
    success: bool = Field(description="Whether the run achieved its goal.")
    status: str = Field(description="Terminal status reported by the execution runtime.")
    reason: str = Field(description="Human-readable terminal reason.")
    error: Optional[str] = Field(default=None, description="Optional execution error.")
    steps: int = Field(ge=0, description="Number of steps executed.")
    finished: datetime = Field(description="Timestamp when recording finishes.")
    elapsed: int = Field(ge=0, description="Elapsed run duration in milliseconds.")
    metadata: Metadata = Field(
        default_factory=Metadata, description="Optional non-critical metadata."
    )


class Projection(BaseModel):
    """
    Request to project scheduled interaction jobs into memory.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    tenant: str = Field(description="Tenant that owns the projection jobs.")
    owner: str = Field(description="Worker identity claiming projection jobs.")
    claimed: datetime = Field(description="Timestamp when jobs are claimed.")
    limit: int = Field(default=10, gt=0, description="Maximum number of jobs to process.")
