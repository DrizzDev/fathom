from __future__ import annotations

from datetime import datetime  # noqa: TC003
from typing import Optional, Tuple

from pydantic import Field, JsonValue

from fathom.constants.collaboration import (
    ActorKind,
    ArtifactKind,
    Audience,
    Label,
    MembershipRole,
    MembershipScope,
    MessageKind,
    ScriptFormat,
    ScriptStatus,
    TaskKind,
    TaskState,
    ThreadState,
)
from fathom.constants.conversation import EntryKind, Visibility
from fathom.constants.signing import SigningStatus
from fathom.schemas.conversation.base import ConversationAliasSchema, ConversationSchema


class ThreadTitleMetadataView(ConversationAliasSchema):
    """
    Metadata describing how a conversation title was produced.
    """

    source: Optional[str] = Field(
        default=None,
        description="Producer that supplied the current title.",
    )
    refreshed: Optional[datetime] = Field(
        default=None,
        validation_alias="refreshed_at",
        serialization_alias="refreshed_at",
        description="Timestamp when the title metadata was refreshed.",
    )


class ThreadMetadataView(ConversationAliasSchema):
    """
    Public metadata grouped under the conversation thread.
    """

    title: Optional[ThreadTitleMetadataView] = Field(
        default=None,
        description="Optional metadata describing how the title was produced.",
    )


class ExecutionReference(ConversationSchema):
    """
    Nested execution identifier for client-facing surfaces.
    """

    id: str = Field(description="Stable execution identifier.")


class WorkflowReference(ConversationSchema):
    """
    Nested workflow identifier for client-facing surfaces.
    """

    id: str = Field(description="Stable workflow identifier.")


class RuntimeReference(ConversationSchema):
    """
    Latest execution and workflow surfaced at the response root of every conversation endpoint.
    """

    execution: Optional[ExecutionReference] = Field(
        default=None,
        description="Latest execution for the conversation.",
    )
    workflow: Optional[WorkflowReference] = Field(
        default=None,
        description="Latest runtime workflow for the conversation.",
    )


class ThreadView(ConversationAliasSchema):
    """
    Client-facing conversation thread summary.
    """

    id: str = Field(description="Stable conversation thread identifier.")
    title: Optional[str] = Field(default=None, description="User-facing thread title.")
    metadata: ThreadMetadataView = Field(
        default_factory=ThreadMetadataView,
        description="Public conversation metadata grouped by concern.",
    )

    state: ThreadState = Field(description="Current thread lifecycle state.")
    digest: Optional[str] = Field(
        default=None,
        description="Optional human-readable conversation digest.",
    )

    created: datetime = Field(
        serialization_alias="created_at", description="Timestamp when the thread was created."
    )
    updated: datetime = Field(
        serialization_alias="updated_at",
        description="Timestamp when the thread was last updated.",
    )


class ActorView(ConversationSchema):
    """
    Client-facing actor summary.
    """

    id: str = Field(description="Stable actor identifier.")
    kind: ActorKind = Field(description="Actor category.")
    name: str = Field(description="User-facing actor name.")
    created: datetime = Field(description="Timestamp when the actor was registered.")


class MemberView(ConversationSchema):
    """
    Client-facing thread membership summary.
    """

    id: str = Field(description="Stable membership identifier.")
    actor: str = Field(description="Actor that joined the thread.")

    scope: MembershipScope = Field(description="Membership scope.")
    joined: datetime = Field(description="Timestamp when the actor joined.")
    role: MembershipRole = Field(description="Actor role inside the thread.")


class TaskNodeView(ConversationAliasSchema):
    """
    Client-facing task tree node.
    """

    id: str = Field(description="Stable task identifier.")
    execution: ExecutionReference = Field(description="Execution that owns the task.")
    workflow: Optional[WorkflowReference] = Field(
        default=None, description="Runtime workflow that produced the task, when known."
    )

    root: Optional[str] = Field(default=None, description="Root task identifier.")
    parent: Optional[str] = Field(default=None, description="Parent task identifier.")

    kind: TaskKind = Field(description="Task category.")
    objective: str = Field(description="Task objective.")
    state: TaskState = Field(description="Task lifecycle state.")

    summary: Optional[str] = Field(default=None, description="Task result summary.")
    assignee: Optional[str] = Field(default=None, description="Actor assigned to the task.")

    created: datetime = Field(
        serialization_alias="created_at", description="Timestamp when the task was created."
    )
    started: Optional[datetime] = Field(
        default=None,
        serialization_alias="started_at",
        description="Timestamp when the task started.",
    )
    ended: Optional[datetime] = Field(
        default=None, serialization_alias="ended_at", description="Timestamp when the task ended."
    )
    children: Tuple[TaskNodeView, ...] = Field(
        default_factory=tuple,
        description="Child tasks nested under this task.",
    )


class TaskTreeView(ConversationAliasSchema):
    """
    Client-facing task hierarchy for one conversation.
    """

    thread: ThreadView = Field(
        serialization_alias="conversation",
        description="Conversation that owns this task tree.",
    )
    runtime: Optional[RuntimeReference] = Field(
        default=None,
        description="Latest execution and workflow for the conversation.",
    )
    roots: Tuple[TaskNodeView, ...] = Field(description="Root task nodes.")
    total: int = Field(ge=0, description="Total task count in the tree.")


class EntryView(ConversationAliasSchema):
    """
    Client-facing timeline entry.
    """

    id: str = Field(description="Stable entry identifier.")
    kind: EntryKind = Field(description="Renderable entry category.")
    visibility: Visibility = Field(description="Visibility classification for rendering.")
    sequence: Optional[int] = Field(default=None, description="Ledger sequence when available.")

    payload: JsonValue = Field(description="JSON-safe render payload.")
    task: Optional[str] = Field(default=None, description="Task associated with the entry.")
    actor: Optional[str] = Field(default=None, description="Actor associated with the entry.")

    created: datetime = Field(
        serialization_alias="created_at", description="Timestamp used for timeline ordering."
    )


class MessageView(ConversationAliasSchema):
    """
    Client-facing message row.
    """

    id: str = Field(description="Stable message identifier.")
    task: Optional[str] = Field(default=None, description="Optional task identifier.")

    author: str = Field(description="Actor that authored the message.")
    reply: Optional[str] = Field(default=None, description="Optional parent message.")

    kind: MessageKind = Field(description="Message kind.")
    audience: Audience = Field(description="Message audience.")

    body: JsonValue = Field(description="JSON-safe message body.")
    sequence: int = Field(ge=0, description="Message sequence inside the thread.")
    labels: Tuple[Label, ...] = Field(description="Policy labels attached to the message.")

    created: datetime = Field(
        serialization_alias="created_at", description="Timestamp when the message was recorded."
    )


class ArtifactView(ConversationAliasSchema):
    """
    Client-facing artifact row.
    """

    id: str = Field(description="Stable artifact identifier.")
    task: Optional[str] = Field(default=None, description="Optional task identifier.")
    producer: Optional[str] = Field(default=None, description="Actor that produced it.")

    kind: ArtifactKind = Field(description="Artifact kind.")
    uri: str = Field(description="Stable artifact location.")
    mime: Optional[str] = Field(default=None, description="Optional media type.")
    size: Optional[int] = Field(default=None, ge=0, description="Artifact size in bytes.")
    labels: Tuple[Label, ...] = Field(description="Policy labels attached to the artifact.")

    created: datetime = Field(
        serialization_alias="created_at", description="Timestamp when the artifact was linked."
    )
    signing_status: SigningStatus = Field(
        default=SigningStatus.NOT_REQUIRED,
        description="Typed signing outcome.",
    )


class ScriptView(ConversationAliasSchema):
    """
    Client-facing reusable script row.
    """

    id: str = Field(description="Stable script identifier.")
    task: Optional[str] = Field(default=None, description="Run-root task identifier when recorded.")

    title: Optional[str] = Field(default=None, description="User-facing script title.")

    format: ScriptFormat = Field(description="Script content format.")
    status: ScriptStatus = Field(description="Script document lifecycle state.")
    revision: int = Field(ge=1, description="Latest immutable version number.")

    checksum: Optional[str] = Field(
        default=None,
        description="SHA-256 of the latest version content; null when the caller skipped the lookup.",
    )

    size: int = Field(ge=0, description="Content size in bytes.")
    content: str = Field(description="Inline current editable content.")

    created_by: Optional[str] = Field(default=None, description="Actor that created the script.")
    updated_by: Optional[str] = Field(
        default=None, description="Actor that last updated the script."
    )
    created: datetime = Field(
        serialization_alias="created_at",
        description="Timestamp when the script was first persisted.",
    )
    updated: datetime = Field(
        serialization_alias="updated_at", description="Timestamp when the script was last updated."
    )


class TimelineView(ConversationAliasSchema):
    """
    Client-facing timeline response for one conversation.
    """

    thread: ThreadView = Field(
        serialization_alias="conversation",
        description="Conversation rendered by this timeline.",
    )
    runtime: Optional[RuntimeReference] = Field(
        default=None,
        description="Latest execution and workflow for the conversation.",
    )
    entries: Tuple[EntryView, ...] = Field(description="Renderable timeline entries.")
    total: int = Field(
        ge=0,
        description=(
            "Sum of matching ledger rows across all included kinds, before visibility "
            "filtering. Used as a coarse hint for clients sizing pagination affordance's."
        ),
    )
    next: Optional[str] = Field(default=None, description="Opaque cursor for the next page.")
