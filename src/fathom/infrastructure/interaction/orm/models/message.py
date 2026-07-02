from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from pydantic import JsonValue
from tortoise.fields import (
    NO_ACTION,
    BigIntField,
    CharField,
    DatetimeField,
    Field,
    ForeignKeyField,
    ForeignKeyRelation,
    JSONField,
)

from fathom.infrastructure.interaction.orm.models.base import Mutable, Record

if TYPE_CHECKING:
    from datetime import datetime

    from fathom.infrastructure.interaction.orm.models.conversation import ConversationRecord
    from fathom.infrastructure.interaction.orm.models.execution import ExecutionRecord


class MessageRecord(Mutable, Record):
    """
    Persistent model for durable conversation messages.
    """

    if TYPE_CHECKING:
        execution_id: str
        conversation_id: str

    conversation: ForeignKeyRelation[ConversationRecord] = ForeignKeyField(
        db_index=True,
        related_name=False,
        db_constraint=False,
        on_delete=NO_ACTION,
        to="models.ConversationRecord",
        source_field="conversation_id",
        description="Conversation that owns the message.",
    )
    execution: ForeignKeyRelation[ExecutionRecord] = ForeignKeyField(
        related_name=False,
        db_constraint=False,
        on_delete=NO_ACTION,
        to="models.ExecutionRecord",
        source_field="execution_id",
        description="Execution that owns the message.",
    )

    task_id: Field[Optional[str]] = CharField(
        null=True,
        max_length=36,
        description="Optional task identifier for the message.",
    )
    author: Field[str] = CharField(
        max_length=128,
        description="Actor that authored the message.",
    )
    reply_id: Field[Optional[str]] = CharField(
        null=True,
        max_length=36,
        description="Optional parent message identifier.",
    )
    sequence: Field[int] = BigIntField(
        description="Positive per-conversation message sequence.",
    )
    kind: Field[str] = CharField(
        max_length=64,
        description="Message category.",
    )
    audience: Field[JsonValue] = JSONField(
        default=list,
        description="Typed audience list for routing.",
    )
    body: Field[JsonValue] = JSONField(
        description="Typed message body payload.",
    )
    labels: Field[JsonValue] = JSONField(
        default=list,
        description="Typed labels attached to the message.",
    )
    sanitized_at: Field[Optional[datetime]] = DatetimeField(
        null=True,
        description="Time when message content was sanitized.",
    )
    sanitizer: Field[Optional[str]] = CharField(
        null=True,
        max_length=128,
        description="Sanitizer profile used on the message.",
    )

    class Meta(Record.Meta):
        abstract = False
        table = "messages"
        table_description = "Conversation message stream with optional execution context."

        indexes = (
            ("tenant_id", "deleted_at"),
            ("tenant_id", "task_id", "sequence"),
            ("tenant_id", "execution_id", "sequence"),
            ("tenant_id", "conversation_id", "sequence"),
        )
        unique_together = (("tenant_id", "conversation_id", "sequence"),)
