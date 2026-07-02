from __future__ import annotations

from typing import Tuple

from pydantic import Field

from fathom.schemas.conversation.base import PageEnvelope
from fathom.schemas.conversation.views import (
    ArtifactView,
    MessageView,
    ScriptView,
    ThreadView,
)


class ConversationPage(PageEnvelope):
    """
    Client-facing paginated conversation list.
    """

    items: Tuple[ThreadView, ...] = Field(description="Conversations in page order.")


class MessagePage(PageEnvelope):
    """
    Client-facing paginated message list.
    """

    items: Tuple[MessageView, ...] = Field(description="Messages in page order.")


class ArtifactPage(PageEnvelope):
    """
    Client-facing paginated artifact list.
    """

    items: Tuple[ArtifactView, ...] = Field(description="Artifacts in page order.")


class ScriptPage(PageEnvelope):
    """
    Client-facing paginated script list.
    """

    items: Tuple[ScriptView, ...] = Field(description="Scripts in page order.")
