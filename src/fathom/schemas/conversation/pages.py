from __future__ import annotations

from typing import Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field

from fathom.schemas.conversation.views import (
    ArtifactView,
    MessageView,
    ScriptView,
    ThreadView,
)


class ConversationPage(BaseModel):
    """
    Client-facing paginated conversation list.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    items: Tuple[ThreadView, ...] = Field(description="Conversations in page order.")
    next: Optional[str] = Field(default=None, description="Opaque next-page cursor.")
    total: int = Field(ge=0, description="Total conversations matching the query.")


class MessagePage(BaseModel):
    """
    Client-facing paginated message list.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    items: Tuple[MessageView, ...] = Field(description="Messages in page order.")
    next: Optional[str] = Field(default=None, description="Opaque next-page cursor.")
    total: int = Field(ge=0, description="Total messages matching the query.")


class ArtifactPage(BaseModel):
    """
    Client-facing paginated artifact list.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    items: Tuple[ArtifactView, ...] = Field(description="Artifacts in page order.")
    next: Optional[str] = Field(default=None, description="Opaque next-page cursor.")
    total: int = Field(ge=0, description="Total artifacts matching the query.")


class ScriptPage(BaseModel):
    """
    Client-facing paginated script list.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    total: int = Field(ge=0, description="Total scripts matching the query.")
    items: Tuple[ScriptView, ...] = Field(description="Scripts in page order.")
    next: Optional[str] = Field(default=None, description="Opaque next-page cursor.")
