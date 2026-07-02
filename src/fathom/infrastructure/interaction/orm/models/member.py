from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from tortoise.fields import (
    NO_ACTION,
    CharField,
    DatetimeField,
    Field,
    ForeignKeyField,
    ForeignKeyRelation,
)

from fathom.infrastructure.interaction.orm.models.base import Mutable, Record

if TYPE_CHECKING:
    from datetime import datetime

    from fathom.infrastructure.interaction.orm.models.conversation import ConversationRecord


class MembershipRecord(Mutable, Record):
    """
    Persistent model for actor membership in a conversation.
    """

    if TYPE_CHECKING:
        conversation_id: str

    conversation: ForeignKeyRelation[ConversationRecord] = ForeignKeyField(
        db_index=True,
        related_name=False,
        db_constraint=False,
        on_delete=NO_ACTION,
        to="models.ConversationRecord",
        source_field="conversation_id",
        description="Conversation that owns the membership.",
    )

    actor: Field[str] = CharField(
        db_index=True,
        max_length=128,
        description="Actor granted membership.",
    )
    role: Field[str] = CharField(
        max_length=64,
        description="Membership role.",
    )
    scope: Field[str] = CharField(
        max_length=64,
        description="Membership scope.",
    )

    joined_at: Field[datetime] = DatetimeField(
        description="Time when the actor joined the conversation.",
    )
    departed_at: Field[Optional[datetime]] = DatetimeField(
        null=True,
        description="Time when the actor left the conversation.",
    )

    class Meta(Record.Meta):
        abstract = False
        table = "memberships"
        table_description = "Actor membership and access grant for one conversation."

        indexes = (
            ("tenant_id", "actor"),
            ("tenant_id", "conversation_id"),
        )
