from __future__ import annotations

from datetime import datetime  # noqa: TC003
from typing import Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field

from fathom.constants.collaboration import ArtifactKind, MessageKind
from fathom.constants.conversation import (
    ARTIFACT_LIST_DEFAULT_LIMIT,
    ARTIFACT_LIST_MAX_LIMIT,
    CONVERSATION_LIST_DEFAULT_LIMIT,
    CONVERSATION_LIST_MAX_LIMIT,
    MESSAGE_LIST_DEFAULT_LIMIT,
    MESSAGE_LIST_MAX_LIMIT,
    SCRIPT_LIST_DEFAULT_LIMIT,
    SCRIPT_LIST_MAX_LIMIT,
    THREAD_TITLE_PREFIX_MAX_LENGTH,
    TIMELINE_DEFAULT_LIMIT,
    TIMELINE_MAX_LIMIT,
    EntryKind,
    Visibility,
)
from fathom.schemas.interaction import SortOrder


class ConversationThreadQuery(BaseModel):
    """
    Query for loading one client-facing conversation thread.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    thread: str = Field(description="Conversation thread identifier.")
    tenant: str = Field(description="Tenant that owns the conversation thread.")


class ConversationTransition(BaseModel):
    """
    Command for archive, unarchive, and soft-delete conversation operations.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    thread: str = Field(description="Conversation thread identifier.")
    tenant: str = Field(description="Tenant that owns the conversation thread.")
    actor: Optional[str] = Field(default=None, description="Actor that requested the transition.")

    updated: datetime = Field(description="Timestamp of the host-issued lifecycle update.")


class TaskTreeQuery(BaseModel):
    """
    Query for rendering one conversation task tree.

    Pass `task` to render a subtree rooted at that task; omit to render every root-level task in the thread.
    Subtree queries are bounded and remain cheap on conversations with many runs.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    thread: str = Field(description="Conversation thread identifier.")
    tenant: str = Field(description="Tenant that owns the conversation task tree.")
    task: Optional[str] = Field(
        default=None,
        description="Optional root-task identifier; when set, restrict the tree to that subtree.",
    )


class TimelineQuery(BaseModel):
    """
    Query for rendering a conversation timeline.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    tenant: str = Field(description="Tenant that owns the conversation timeline.")

    thread: str = Field(description="Conversation thread identifier.")
    task: Optional[str] = Field(default=None, description="Optional task filter.")
    actor: Optional[str] = Field(
        default=None,
        description=(
            "Optional actor filter. Matches the role-specific identity in each kind: "
            "message author, event actor, artifact producer, context consumer."
        ),
    )
    kinds: Tuple[EntryKind, ...] = Field(
        default_factory=tuple,
        description=(
            "Optional timeline entry-kind filter. Applied after the visibility mode filter."
        ),
    )
    since: Optional[datetime] = Field(
        default=None,
        description="Only include entries created at or after this timestamp.",
    )
    until: Optional[datetime] = Field(
        default=None,
        description="Only include entries created before this timestamp.",
    )
    mode: Visibility = Field(default=Visibility.USER, description="Timeline visibility mode.")
    limit: int = Field(
        gt=0,
        le=TIMELINE_MAX_LIMIT,
        default=TIMELINE_DEFAULT_LIMIT,
        description="Maximum entries to return.",
    )
    cursor: Optional[str] = Field(default=None, description="Opaque pagination cursor.")
    order: SortOrder = Field(
        default=SortOrder.DESC,
        description=(
            "Sort direction by created timestamp (DESC = newest first, ASC = oldest first). "
            "DESC is the default so chat-style clients render newest at the bottom and scroll up to fetch older. Cursor payload is direction-agnostic; callers must keep `order` consistent across pages or pagination will skip/repeat rows."
        ),
    )


class ConversationListQuery(BaseModel):
    """
    Query for listing client-facing conversations.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    tenant: str = Field(description="Tenant that owns the conversations.")
    workspace: Optional[str] = Field(default=None, description="Optional workspace filter.")

    state: Optional[str] = Field(default=None, description="Optional thread state filter.")
    since: Optional[datetime] = Field(
        default=None,
        description="Only include conversations updated at or after this timestamp.",
    )
    until: Optional[datetime] = Field(
        default=None,
        description="Only include conversations updated before this timestamp.",
    )
    title: Optional[str] = Field(
        default=None,
        max_length=THREAD_TITLE_PREFIX_MAX_LENGTH,
        description="Optional case-insensitive prefix match against conversation titles.",
    )
    limit: int = Field(
        gt=0,
        le=CONVERSATION_LIST_MAX_LIMIT,
        default=CONVERSATION_LIST_DEFAULT_LIMIT,
        description="Maximum conversations to return.",
    )
    cursor: Optional[str] = Field(default=None, description="Opaque pagination cursor.")


class MessageListQuery(BaseModel):
    """
    Query for listing messages in one conversation.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    tenant: str = Field(description="Tenant that owns the messages.")
    thread: str = Field(description="Conversation thread identifier.")
    task: Optional[str] = Field(default=None, description="Optional task filter.")
    actor: Optional[str] = Field(default=None, description="Optional author filter.")

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
    limit: int = Field(
        gt=0,
        le=MESSAGE_LIST_MAX_LIMIT,
        default=MESSAGE_LIST_DEFAULT_LIMIT,
        description="Maximum messages to return.",
    )
    cursor: Optional[str] = Field(default=None, description="Opaque pagination cursor.")
    order: SortOrder = Field(
        default=SortOrder.DESC,
        description=(
            "Sort direction by created timestamp (DESC = newest first, ASC = oldest first). "
            "DESC is the default so chat-style clients render newest at the bottom and scroll up to fetch older. Cursor payload is direction-agnostic; callers must keep `order` consistent across pages or pagination will skip/repeat rows."
        ),
    )


class ArtifactListQuery(BaseModel):
    """
    Query for listing artifacts in one conversation.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    tenant: str = Field(description="Tenant that owns the artifacts.")
    thread: str = Field(description="Conversation thread identifier.")
    task: Optional[str] = Field(default=None, description="Optional task filter.")
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
    limit: int = Field(
        gt=0,
        le=ARTIFACT_LIST_MAX_LIMIT,
        default=ARTIFACT_LIST_DEFAULT_LIMIT,
        description="Maximum artifacts to return.",
    )
    cursor: Optional[str] = Field(default=None, description="Opaque pagination cursor.")
    order: SortOrder = Field(
        default=SortOrder.DESC,
        description=(
            "Sort direction by created timestamp (DESC = newest first, ASC = oldest first). "
            "DESC is the default so chat-style clients render newest at the bottom. Cursor payload is direction-agnostic; callers must keep `order` consistent across pages or pagination will skip/repeat rows."
        ),
    )


class RunScriptQuery(BaseModel):
    """
    Query for loading the generated script for one run task.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    task: str = Field(description="Run root task identifier.")
    tenant: str = Field(description="Tenant that owns the script.")
    thread: str = Field(description="Conversation thread identifier.")


class ScriptsQuery(BaseModel):
    """
    Query for listing scripts in one conversation thread.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    tenant: str = Field(description="Tenant that owns the scripts.")
    thread: str = Field(description="Conversation thread identifier.")
    task: Optional[str] = Field(default=None, description="Optional run-root task filter.")

    since: Optional[datetime] = Field(
        default=None,
        description="Only include scripts updated at or after this timestamp.",
    )
    until: Optional[datetime] = Field(
        default=None,
        description="Only include scripts updated before this timestamp.",
    )
    limit: int = Field(
        gt=0,
        le=SCRIPT_LIST_MAX_LIMIT,
        default=SCRIPT_LIST_DEFAULT_LIMIT,
        description="Maximum scripts to return.",
    )
    cursor: Optional[str] = Field(default=None, description="Opaque pagination cursor.")
    count: bool = Field(
        default=True,
        description="Whether to run COUNT(*) for the total match estimate; false skips the scan.",
    )
