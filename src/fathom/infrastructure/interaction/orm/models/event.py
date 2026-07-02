from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from pydantic import JsonValue
from tortoise.fields import (
    NO_ACTION,
    BigIntField,
    CharField,
    Field,
    ForeignKeyField,
    ForeignKeyNullableRelation,
    ForeignKeyRelation,
    JSONField,
)

from fathom.infrastructure.interaction.orm.models.base import Mutable, Record

if TYPE_CHECKING:
    from fathom.infrastructure.interaction.orm.models.conversation import ConversationRecord
    from fathom.infrastructure.interaction.orm.models.execution import ExecutionRecord


class EventRecord(Mutable, Record):
    """
    Persistent model for append-only conversation lifecycle events.
    """

    if TYPE_CHECKING:
        conversation_id: str
        execution_id: Optional[str]

    conversation: ForeignKeyRelation[ConversationRecord] = ForeignKeyField(
        db_index=True,
        related_name=False,
        db_constraint=False,
        on_delete=NO_ACTION,
        to="models.ConversationRecord",
        source_field="conversation_id",
        description="Conversation that owns the event.",
    )
    execution: ForeignKeyNullableRelation[ExecutionRecord] = ForeignKeyField(
        null=True,
        related_name=False,
        db_constraint=False,
        on_delete=NO_ACTION,
        to="models.ExecutionRecord",
        source_field="execution_id",
        description="Optional execution context for the event.",
    )
    task_id: Field[Optional[str]] = CharField(
        null=True,
        max_length=36,
        description="Optional task identifier for the event.",
    )
    actor: Field[Optional[str]] = CharField(
        null=True,
        max_length=128,
        description="Actor associated with the event.",
    )
    sequence: Field[int] = BigIntField(
        description="Positive per-conversation event sequence.",
    )
    kind: Field[str] = CharField(
        max_length=128,
        description="Event category.",
    )
    source: Field[str] = CharField(
        max_length=128,
        description="System component that produced the event.",
    )
    payload: Field[JsonValue] = JSONField(
        description="Typed event payload.",
    )

    class Meta(Record.Meta):
        abstract = False
        table = "events"
        table_description = "Append-only event stream for conversation lifecycle changes."

        indexes = (
            ("tenant_id", "task_id", "sequence"),
            ("tenant_id", "execution_id", "sequence"),
            ("tenant_id", "conversation_id", "sequence"),
        )
        unique_together = (("tenant_id", "conversation_id", "sequence"),)
