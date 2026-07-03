from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from tortoise.fields import CharField, DatetimeField, Field

from fathom.infrastructure.interaction.orm.models.base import Mutable, Record

if TYPE_CHECKING:
    from datetime import datetime


class ConversationRecord(Mutable, Record):
    """
    Persistent model for client-facing conversation containers.
    """

    title: Field[Optional[str]] = CharField(
        null=True,
        max_length=1024,
        description="Display title shown in conversation lists.",
    )
    digest: Field[Optional[str]] = CharField(
        null=True,
        max_length=128,
        description="Latest digest for the visible conversation state.",
    )
    archived_at: Field[Optional[datetime]] = DatetimeField(
        null=True,
        description="Time when the conversation was archived.",
    )
    archived_by: Field[Optional[str]] = CharField(
        null=True,
        max_length=128,
        description="Actor or operator that archived the conversation.",
    )

    class Meta(Record.Meta):
        abstract = False
        table = "conversations"
        table_description = "Client-facing conversation container scoped by tenant."

        indexes = (
            ("tenant_id", "updated_at"),
            ("tenant_id", "deleted_at"),
            ("tenant_id", "archived_at"),
        )
