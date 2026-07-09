from __future__ import annotations

from datetime import datetime  # noqa: TC003
from typing import Dict, Optional, Tuple

from pydantic import Field, JsonValue, model_validator

from fathom.constants.collaboration import (
    ActorKind,
    ArtifactBackend,
    ArtifactKind,
    Audience,
    ContextPurpose,
    Label,
    MembershipRole,
    MembershipScope,
    MessageKind,
    ScriptFormat,
    ScriptStatus,
    ScriptVersionSource,
    TaskCode,
    TaskKind,
    TaskState,
)
from fathom.constants.conversation import THREAD_TITLE_INPUT_MAX_LENGTH, THREAD_TITLE_MAX_LENGTH
from fathom.schemas.conversation.base import (
    ConversationSchema,
    ThreadMetadataScope,
    WorkspaceMetadataScope,
)


class ActorInput(ConversationSchema):
    """
    Actor identity supplied by a host or runtime.
    """

    id: str = Field(description="Stable actor identifier.")
    kind: ActorKind = Field(description="Actor category.")
    name: str = Field(description="User-facing actor name.")
    workspace: Optional[str] = Field(default=None, description="Optional workspace boundary.")
    model: Optional[str] = Field(default=None, description="Optional runtime model reference.")
    provider: Optional[str] = Field(default=None, description="Optional runtime provider name.")


class AddActor(WorkspaceMetadataScope):
    """
    Request to register an actor for conversation participation.
    """

    id: str = Field(description="Stable actor identifier.")
    kind: ActorKind = Field(description="Actor category.")
    name: str = Field(description="User-facing actor name.")
    external: Optional[str] = Field(default=None, description="Optional external reference.")
    model: Optional[str] = Field(default=None, description="Optional runtime model reference.")
    provider: Optional[str] = Field(default=None, description="Optional runtime provider name.")

    created: datetime = Field(description="Timestamp when the actor is registered.")


class JoinMember(ThreadMetadataScope):
    """
    Request to join an actor to a conversation thread.
    """

    id: str = Field(description="Stable membership identifier.")
    actor: str = Field(description="Actor joining the thread.")
    role: MembershipRole = Field(description="Actor role inside the thread.")
    scope: MembershipScope = Field(default=MembershipScope.THREAD, description="Membership scope.")

    joined: datetime = Field(description="Timestamp when the actor joins.")


class ThreadCreate(WorkspaceMetadataScope):
    """
    Request to create a client-facing conversation thread.
    """

    id: str = Field(description="Stable conversation thread identifier.")
    title: Optional[str] = Field(
        default=None,
        max_length=THREAD_TITLE_INPUT_MAX_LENGTH,
        description="User-facing thread title.",
    )
    creator: Optional[ActorInput] = Field(default=None, description="Optional thread creator.")

    member: Optional[str] = Field(
        default=None,
        description="Stable membership identifier for the creator.",
    )
    role: MembershipRole = Field(
        default=MembershipRole.OWNER,
        description="Creator role inside the thread.",
    )
    created: datetime = Field(description="Timestamp when the thread is created.")

    @model_validator(mode="after")
    def require_creator_membership(self) -> ThreadCreate:
        """
        Require an explicit membership identifier when a creator is supplied.
        """

        if self.creator is not None and self.member is None:
            raise ValueError("Thread creator requires a stable membership identifier.")

        return self


class MessageAppend(ThreadMetadataScope):
    """
    Request to append a message to a conversation thread.
    """

    id: str = Field(description="Stable message identifier.")
    execution: Optional[str] = Field(
        default=None,
        description="Optional execution identifier that owns the message.",
    )
    task: Optional[str] = Field(default=None, description="Optional task identifier.")

    author: str = Field(description="Actor that authored the message.")
    reply: Optional[str] = Field(default=None, description="Optional parent message.")

    sequence: Optional[int] = Field(
        default=None,
        ge=1,
        description=(
            "Optional caller-supplied sequence. None lets the store allocate "
            "the next per-thread sequence atomically; an integer >= 1 is treated as a deterministic caller-owned value."
        ),
    )
    kind: MessageKind = Field(description="Message category.")
    audience: Audience = Field(default=Audience.THREAD, description="Intended audience.")

    body: JsonValue = Field(description="JSON-safe message body.")
    labels: Tuple[Label, ...] = Field(
        default_factory=tuple, description="Policy labels attached to the message."
    )
    created: datetime = Field(description="Timestamp when the message is appended.")


class TaskStart(ThreadMetadataScope):
    """
    Request to start a task inside a conversation thread.
    """

    id: str = Field(description="Stable task identifier.")
    execution: str = Field(description="Execution identifier that owns the task.")
    creator: Optional[str] = Field(default=None, description="Actor that created the task.")
    assignee: Optional[str] = Field(default=None, description="Actor assigned to the task.")
    parent: Optional[str] = Field(default=None, description="Optional parent task identifier.")

    root: Optional[str] = Field(default=None, description="Optional root task identifier.")
    origin: Optional[str] = Field(default=None, description="Optional origin message identifier.")

    kind: TaskKind = Field(description="Task category.")
    state: TaskState = Field(default=TaskState.RUNNING, description="Initial task state.")

    objective: str = Field(description="Human-readable task objective.")
    reference: Optional[str] = Field(default=None, description="Optional target reference.")

    plan: Dict[str, JsonValue] = Field(
        default_factory=dict,
        description="Structured plan data supplied by the runtime.",
    )
    progress: Dict[str, JsonValue] = Field(
        default_factory=dict,
        description="Structured progress data supplied by the runtime.",
    )

    created: datetime = Field(description="Timestamp when the task is started.")


class TaskFinish(ConversationSchema):
    """
    Request to finish a conversation task.
    """

    tenant: str = Field(description="Tenant that owns the task.")
    task: str = Field(description="Task identifier to finish.")
    state: TaskState = Field(description="Terminal task state.")
    code: TaskCode = Field(description="Machine-readable terminal task code.")

    summary: Optional[str] = Field(default=None, description="Task result summary.")
    detail: Optional[str] = Field(default=None, description="Human-readable terminal detail.")

    ended: datetime = Field(description="Timestamp when the task is finished.")
    elapsed: int = Field(ge=0, description="Elapsed task duration in milliseconds.")


class ArtifactAttach(ThreadMetadataScope):
    """
    Request to attach an artifact reference to a conversation thread.
    """

    id: str = Field(description="Stable artifact identifier.")
    execution: Optional[str] = Field(
        default=None,
        description="Optional execution identifier that owns the artifact.",
    )
    task: Optional[str] = Field(default=None, description="Optional task identifier.")
    producer: Optional[str] = Field(default=None, description="Actor that produced the artifact.")

    uri: str = Field(description="Stable artifact location.")
    kind: ArtifactKind = Field(description="Artifact category.")
    backend: ArtifactBackend = Field(description="Artifact storage backend.")
    retention: Optional[str] = Field(default=None, description="Retention class.")

    mime: Optional[str] = Field(default=None, description="Optional media type.")
    size: Optional[int] = Field(default=None, ge=0, description="Artifact size in bytes.")

    labels: Tuple[Label, ...] = Field(
        default_factory=tuple, description="Policy labels attached to the artifact."
    )
    created: datetime = Field(description="Timestamp when the artifact is attached.")


class ScriptSave(ThreadMetadataScope):
    """
    Request to save a reusable script and version audit row.
    """

    id: str = Field(description="Stable script identifier.")
    execution: Optional[str] = Field(
        default=None,
        description="Optional execution identifier that owns the script.",
    )
    task: Optional[str] = Field(default=None, description="Task that produced the script.")
    artifact: Optional[str] = Field(default=None, description="Export artifact identifier.")

    title: Optional[str] = Field(
        default=None,
        max_length=THREAD_TITLE_MAX_LENGTH,
        description="User-facing script title.",
    )
    format: ScriptFormat = Field(
        default=ScriptFormat.TEXT_PLAIN, description="Script content format."
    )

    status: ScriptStatus = Field(default=ScriptStatus.ACTIVE, description="Script state.")
    content: str = Field(description="Editable script content.")
    source: ScriptVersionSource = Field(
        default=ScriptVersionSource.GENERATED, description="Source of this version."
    )

    actor: Optional[str] = Field(default=None, description="Actor saving the script.")
    summary: Optional[str] = Field(default=None, description="Change summary for audit.")

    created: datetime = Field(description="Timestamp when the script is saved.")


class ContextRecord(ThreadMetadataScope):
    """
    Request to record a reference-based context recipe.
    """

    id: str = Field(description="Stable context identifier.")
    execution: Optional[str] = Field(
        default=None,
        description="Optional execution identifier that owns the context.",
    )
    task: Optional[str] = Field(default=None, description="Optional task identifier.")
    consumer: Optional[str] = Field(default=None, description="Actor that consumes the context.")

    builder: str = Field(description="Builder name and version.")
    purpose: ContextPurpose = Field(description="Context purpose.")

    hash: Optional[str] = Field(default=None, description="Optional stable context hash.")
    events: Tuple[str, ...] = Field(default_factory=tuple, description="Event references.")
    messages: Tuple[str, ...] = Field(default_factory=tuple, description="Message references.")
    artifacts: Tuple[str, ...] = Field(default_factory=tuple, description="Artifact references.")

    model: Optional[str] = Field(default=None, description="Optional model reference.")
    provider: Optional[str] = Field(default=None, description="Optional model provider name.")

    created: datetime = Field(description="Timestamp when the context is recorded.")
    expires: Optional[datetime] = Field(default=None, description="Optional expiry timestamp.")
