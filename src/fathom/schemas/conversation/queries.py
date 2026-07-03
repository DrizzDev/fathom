from __future__ import annotations

from datetime import datetime  # noqa: TC003
from typing import Optional, Tuple

from pydantic import Field

from fathom.constants.collaboration import ArtifactKind, MessageKind, ThreadState
from fathom.constants.conversation import (
    ARTIFACT_LIST_DEFAULT_LIMIT,
    ARTIFACT_LIST_MAX_LIMIT,
    CONVERSATION_LIST_DEFAULT_LIMIT,
    CONVERSATION_LIST_MAX_LIMIT,
    MESSAGE_LIST_DEFAULT_LIMIT,
    MESSAGE_LIST_MAX_LIMIT,
    SCRIPT_LIST_DEFAULT_LIMIT,
    SCRIPT_LIST_MAX_LIMIT,
    TASK_TREE_ROOTS_MAX_LIMIT,
    THREAD_TITLE_PREFIX_MAX_LENGTH,
    TIMELINE_DEFAULT_LIMIT,
    TIMELINE_MAX_LIMIT,
    EntryKind,
    Visibility,
)
from fathom.schemas.conversation.base import (
    CursorScope,
    TenantAccessScope,
    ThreadAccessScope,
    ThreadActorScope,
    TimeWindow,
)
from fathom.schemas.interaction import SortOrder


class ConversationThreadQuery(ThreadAccessScope):
    """
    Query for loading one client-facing conversation thread.
    """


class ConversationTransition(ThreadActorScope):
    """
    Command for archive, unarchive, and soft-delete conversation operations.
    """

    include_archived: bool = Field(
        default=False,
        description="Whether access validation may load an archived conversation.",
    )

    updated: datetime = Field(description="Timestamp of the host-issued lifecycle update.")


class TaskTreeQuery(ThreadAccessScope):
    """
    Query for rendering one conversation task tree.

    Pass `task` to render a subtree rooted at that task; omit to render every root-level task in the thread.
    Subtree queries are bounded and remain cheap on conversations with many runs.
    """

    task: Optional[str] = Field(
        default=None,
        description="Optional root-task identifier; when set, restrict the tree to that subtree.",
    )
    limit: int = Field(
        gt=0,
        le=TASK_TREE_ROOTS_MAX_LIMIT,
        default=TASK_TREE_ROOTS_MAX_LIMIT,
        description="Server-side cap on the number of root nodes returned; children are unbounded.",
    )


class TimelineQuery(ThreadAccessScope, TimeWindow, CursorScope):
    """
    Query for rendering a conversation timeline.
    """

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
    mode: Visibility = Field(default=Visibility.USER, description="Timeline visibility mode.")
    limit: int = Field(
        gt=0,
        le=TIMELINE_MAX_LIMIT,
        default=TIMELINE_DEFAULT_LIMIT,
        description="Maximum entries to return.",
    )
    order: SortOrder = Field(
        default=SortOrder.DESC,
        description=(
            "Sort direction by created timestamp (DESC = newest first, ASC = oldest first). "
            "DESC is the default so chat-style clients render newest at the bottom and scroll up to fetch older. Cursor payload is direction-agnostic; callers must keep `order` consistent across pages or pagination will skip/repeat rows."
        ),
    )


class ConversationListQuery(TenantAccessScope, TimeWindow, CursorScope):
    """
    Query for listing client-facing conversations.
    """

    workspace: Optional[str] = Field(default=None, description="Optional workspace filter.")

    state: Optional[ThreadState] = Field(
        default=None,
        description="Optional thread state filter.",
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


class MessageListQuery(ThreadAccessScope, TimeWindow, CursorScope):
    """
    Query for listing messages in one conversation.
    """

    task: Optional[str] = Field(default=None, description="Optional task filter.")
    actor: Optional[str] = Field(default=None, description="Optional author filter.")

    kinds: Tuple[MessageKind, ...] = Field(
        default_factory=tuple,
        description="Optional message-kind filter.",
    )
    limit: int = Field(
        gt=0,
        le=MESSAGE_LIST_MAX_LIMIT,
        default=MESSAGE_LIST_DEFAULT_LIMIT,
        description="Maximum messages to return.",
    )
    order: SortOrder = Field(
        default=SortOrder.DESC,
        description=(
            "Sort direction by created timestamp (DESC = newest first, ASC = oldest first). "
            "DESC is the default so chat-style clients render newest at the bottom and scroll up to fetch older. Cursor payload is direction-agnostic; callers must keep `order` consistent across pages or pagination will skip/repeat rows."
        ),
    )


class ArtifactListQuery(ThreadAccessScope, TimeWindow, CursorScope):
    """
    Query for listing artifacts in one conversation.
    """

    task: Optional[str] = Field(default=None, description="Optional task filter.")
    producer: Optional[str] = Field(default=None, description="Optional producer filter.")

    kinds: Tuple[ArtifactKind, ...] = Field(
        default_factory=tuple,
        description="Optional artifact-kind filter.",
    )
    limit: int = Field(
        gt=0,
        le=ARTIFACT_LIST_MAX_LIMIT,
        default=ARTIFACT_LIST_DEFAULT_LIMIT,
        description="Maximum artifacts to return.",
    )
    order: SortOrder = Field(
        default=SortOrder.DESC,
        description=(
            "Sort direction by created timestamp (DESC = newest first, ASC = oldest first). "
            "DESC is the default so chat-style clients render newest at the bottom. Cursor payload is direction-agnostic; callers must keep `order` consistent across pages or pagination will skip/repeat rows."
        ),
    )


class RunScriptQuery(ThreadAccessScope):
    """
    Query for loading the generated script for one run task.
    """

    task: str = Field(description="Run root task identifier.")


class ScriptsQuery(ThreadAccessScope, TimeWindow, CursorScope):
    """
    Query for listing scripts in one conversation thread.
    """

    task: Optional[str] = Field(default=None, description="Optional run-root task filter.")

    limit: int = Field(
        gt=0,
        le=SCRIPT_LIST_MAX_LIMIT,
        default=SCRIPT_LIST_DEFAULT_LIMIT,
        description="Maximum scripts to return.",
    )
    count: bool = Field(
        default=True,
        description="Whether to run COUNT(*) for the total match estimate; false skips the scan.",
    )


class SummaryQuery(ThreadAccessScope):
    """
    Query for projecting one client-facing conversation summary.
    """
