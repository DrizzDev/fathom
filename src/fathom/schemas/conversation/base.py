from __future__ import annotations

from datetime import datetime  # noqa: TC003
from typing import Dict, Optional

from pydantic import BaseModel, ConfigDict, Field, JsonValue


class ConversationSchema(BaseModel):
    """
    Base model for immutable conversation boundary schemas.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")


class ConversationAliasSchema(ConversationSchema):
    """
    Base model for conversation schemas that expose serialization aliases.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)


class SummaryBodySchema(BaseModel):
    """
    Base model for permissive summary source message bodies.
    """

    model_config = ConfigDict(extra="ignore", populate_by_name=True)


class TenantScope(ConversationSchema):
    """
    Shared tenant boundary for conversation schemas.
    """

    tenant: str = Field(description="Tenant that owns the record.")


class WorkspaceScope(TenantScope):
    """
    Shared tenant and workspace boundary for conversation schemas.
    """

    workspace: Optional[str] = Field(default=None, description="Optional workspace boundary.")


class ThreadScope(WorkspaceScope):
    """
    Shared tenant, workspace, and conversation boundary for conversation schemas.
    """

    thread: str = Field(description="Conversation thread identifier.")


class TenantAccessScope(TenantScope):
    """
    Shared tenant and operator boundary for read access queries.
    """

    operator: str = Field(description="Actor requesting access.")


class ThreadAccessScope(TenantAccessScope):
    """
    Shared tenant, conversation, and operator boundary for read access queries.
    """

    thread: str = Field(description="Conversation thread identifier.")


class ThreadActorScope(TenantScope):
    """
    Shared tenant, conversation, and actor boundary for write commands.
    """

    thread: str = Field(description="Conversation thread identifier.")
    actor: str = Field(description="Actor that requested the operation.")


class TimeWindow(ConversationSchema):
    """
    Shared created-or-updated timestamp window for list queries.
    """

    since: Optional[datetime] = Field(
        default=None,
        description="Only include records at or after this timestamp.",
    )
    until: Optional[datetime] = Field(
        default=None,
        description="Only include records before this timestamp.",
    )


class CursorScope(ConversationSchema):
    """
    Shared opaque cursor for paginated query schemas.
    """

    cursor: Optional[str] = Field(default=None, description="Opaque pagination cursor.")


class PageEnvelope(ConversationSchema):
    """
    Shared pagination metadata for conversation list responses.
    """

    next: Optional[str] = Field(default=None, description="Opaque next-page cursor.")
    total: int = Field(ge=0, description="Total records matching the query.")


class WorkspaceMetadataScope(WorkspaceScope):
    """
    Shared tenant, workspace, and metadata boundary for write requests.
    """

    metadata: Dict[str, JsonValue] = Field(
        default_factory=dict,
        description="Optional non-critical metadata.",
    )


class ThreadMetadataScope(ThreadScope):
    """
    Shared conversation and metadata boundary for write requests.
    """

    metadata: Dict[str, JsonValue] = Field(
        default_factory=dict,
        description="Optional non-critical metadata.",
    )
