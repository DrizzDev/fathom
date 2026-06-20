from __future__ import annotations

from datetime import datetime  # noqa: TC003
from typing import Dict, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

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


class ActorInput(BaseModel):
    """
    Actor identity supplied by a host or runtime.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(description="Stable actor identifier.")
    kind: ActorKind = Field(description="Actor category.")
    name: str = Field(description="User-facing actor name.")
    workspace: Optional[str] = Field(default=None, description="Optional workspace boundary.")
    model: Optional[str] = Field(default=None, description="Optional runtime model reference.")
    provider: Optional[str] = Field(default=None, description="Optional runtime provider name.")


class AddActor(BaseModel):
    """
    Request to register an actor for conversation participation.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(description="Stable actor identifier.")
    tenant: str = Field(description="Tenant that owns the actor.")
    workspace: Optional[str] = Field(default=None, description="Optional workspace boundary.")

    kind: ActorKind = Field(description="Actor category.")
    name: str = Field(description="User-facing actor name.")
    external: Optional[str] = Field(default=None, description="Optional external reference.")
    model: Optional[str] = Field(default=None, description="Optional runtime model reference.")
    provider: Optional[str] = Field(default=None, description="Optional runtime provider name.")

    created: datetime = Field(description="Timestamp when the actor is registered.")
    metadata: Dict[str, JsonValue] = Field(
        default_factory=dict,
        description="Optional non-critical actor metadata.",
    )


class JoinMember(BaseModel):
    """
    Request to join an actor to a conversation thread.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(description="Stable membership identifier.")
    thread: str = Field(description="Conversation thread identifier.")
    tenant: str = Field(description="Tenant that owns the membership.")
    workspace: Optional[str] = Field(default=None, description="Optional workspace boundary.")

    actor: str = Field(description="Actor joining the thread.")
    role: MembershipRole = Field(description="Actor role inside the thread.")
    scope: MembershipScope = Field(default=MembershipScope.THREAD, description="Membership scope.")

    joined: datetime = Field(description="Timestamp when the actor joins.")
    metadata: Dict[str, JsonValue] = Field(
        default_factory=dict, description="Optional non-critical membership metadata."
    )


class ThreadCreate(BaseModel):
    """
    Request to create a client-facing conversation thread.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(description="Stable conversation thread identifier.")
    tenant: str = Field(description="Tenant that owns the conversation thread.")
    workspace: Optional[str] = Field(default=None, description="Optional workspace boundary.")

    title: Optional[str] = Field(default=None, description="User-facing thread title.")
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
    metadata: Dict[str, JsonValue] = Field(
        default_factory=dict,
        description="Optional non-critical thread metadata.",
    )

    @model_validator(mode="after")
    def require_creator_membership(self) -> ThreadCreate:
        """
        Require an explicit membership identifier when a creator is supplied.
        """

        if self.creator is not None and self.member is None:
            raise ValueError("Thread creator requires a stable membership identifier.")

        return self


class MessageAppend(BaseModel):
    """
    Request to append a message to a conversation thread.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(description="Stable message identifier.")
    tenant: str = Field(description="Tenant that owns the message.")
    workspace: Optional[str] = Field(default=None, description="Optional workspace boundary.")

    thread: str = Field(description="Conversation thread identifier.")
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
    metadata: Dict[str, JsonValue] = Field(
        default_factory=dict, description="Optional non-critical message metadata."
    )


class TaskStart(BaseModel):
    """
    Request to start a task inside a conversation thread.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(description="Stable task identifier.")
    tenant: str = Field(description="Tenant that owns the task.")
    workspace: Optional[str] = Field(default=None, description="Optional workspace boundary.")

    thread: str = Field(description="Conversation thread identifier.")
    creator: Optional[str] = Field(default=None, description="Actor that created the task.")
    assignee: Optional[str] = Field(default=None, description="Actor assigned to the task.")
    parent: Optional[str] = Field(default=None, description="Optional parent task identifier.")

    root: Optional[str] = Field(default=None, description="Optional root task identifier.")
    origin: Optional[str] = Field(default=None, description="Optional origin message identifier.")

    kind: TaskKind = Field(description="Task category.")
    state: TaskState = Field(default=TaskState.RUNNING, description="Initial task state.")

    objective: str = Field(description="Human-readable task objective.")
    reference: Optional[str] = Field(default=None, description="Optional target reference.")

    created: datetime = Field(description="Timestamp when the task is started.")
    metadata: Dict[str, JsonValue] = Field(
        default_factory=dict, description="Optional non-critical task metadata."
    )


class TaskFinish(BaseModel):
    """
    Request to finish a conversation task.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    tenant: str = Field(description="Tenant that owns the task.")
    task: str = Field(description="Task identifier to finish.")
    state: TaskState = Field(description="Terminal task state.")
    code: TaskCode = Field(description="Machine-readable terminal task code.")

    summary: Optional[str] = Field(default=None, description="Task result summary.")
    detail: Optional[str] = Field(default=None, description="Human-readable terminal detail.")

    ended: datetime = Field(description="Timestamp when the task is finished.")
    elapsed: int = Field(ge=0, description="Elapsed task duration in milliseconds.")


class ArtifactAttach(BaseModel):
    """
    Request to attach an artifact reference to a conversation thread.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(description="Stable artifact identifier.")
    tenant: str = Field(description="Tenant that owns the artifact.")
    workspace: Optional[str] = Field(default=None, description="Optional workspace boundary.")

    thread: str = Field(description="Conversation thread identifier.")
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
    metadata: Dict[str, JsonValue] = Field(
        default_factory=dict, description="Optional non-critical artifact metadata."
    )


class ScriptSave(BaseModel):
    """
    Request to save a reusable script and version audit row.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(description="Stable script identifier.")
    tenant: str = Field(description="Tenant that owns the script.")
    workspace: Optional[str] = Field(default=None, description="Optional workspace boundary.")

    thread: str = Field(description="Conversation thread identifier.")
    task: Optional[str] = Field(default=None, description="Task that produced the script.")
    artifact: Optional[str] = Field(default=None, description="Export artifact identifier.")

    title: Optional[str] = Field(default=None, description="User-facing script title.")
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
    metadata: Dict[str, JsonValue] = Field(
        default_factory=dict,
        description="Optional non-critical script metadata.",
    )


class ContextRecord(BaseModel):
    """
    Request to record a reference-based context recipe.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(description="Stable context identifier.")
    tenant: str = Field(description="Tenant that owns the context.")
    workspace: Optional[str] = Field(default=None, description="Optional workspace boundary.")

    thread: str = Field(description="Conversation thread identifier.")
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
    metadata: Dict[str, JsonValue] = Field(
        default_factory=dict, description="Optional non-critical context metadata."
    )
