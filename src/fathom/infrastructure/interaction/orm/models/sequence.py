from __future__ import annotations

from typing import TYPE_CHECKING

from tortoise.fields import (
    NO_ACTION,
    BigIntField,
    CharField,
    Field,
    ForeignKeyField,
    ForeignKeyRelation,
)

from fathom.infrastructure.interaction.orm.models.base import Mutable, Record

if TYPE_CHECKING:
    from fathom.infrastructure.interaction.orm.models.conversation import ConversationRecord


class SequenceRecord(Mutable, Record):
    """
    Persistent model for per-conversation sequence allocation state.
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
        description="Conversation that owns the sequence.",
    )
    scope: Field[str] = CharField(
        max_length=16,
        description="Sequence scope within the conversation.",
    )
    value: Field[int] = BigIntField(description="Last allocated positive sequence value.")

    class Meta(Record.Meta):
        abstract = False
        table = "sequences"
        table_description = "Internal per-conversation monotonic sequence allocator."

        indexes = (("tenant_id", "conversation_id", "scope"),)
        unique_together = (("tenant_id", "conversation_id", "scope"),)
