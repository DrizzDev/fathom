from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from pydantic import JsonValue
from tortoise.fields import (
    NO_ACTION,
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


class ContextRecord(Mutable, Record):
    """
    Persistent model for reusable context build records.
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
        description="Conversation that owns the context.",
    )
    execution: ForeignKeyRelation[ExecutionRecord] = ForeignKeyField(
        related_name=False,
        db_constraint=False,
        on_delete=NO_ACTION,
        to="models.ExecutionRecord",
        source_field="execution_id",
        description="Execution that owns the context.",
    )

    task_id: Field[Optional[str]] = CharField(
        null=True,
        max_length=36,
        description="Optional task identifier for the context build.",
    )
    consumer: Field[Optional[str]] = CharField(
        null=True,
        max_length=128,
        description="Actor that consumed the context.",
    )

    purpose: Field[str] = CharField(max_length=64, description="Context purpose.")
    builder: Field[str] = CharField(max_length=128, description="Context builder.")

    references: Field[JsonValue] = JSONField(description="Typed context references.")
    budget: Field[JsonValue] = JSONField(default=dict, description="Typed context budget.")
    filters: Field[JsonValue] = JSONField(default=dict, description="Typed context filters.")

    hash: Field[Optional[str]] = CharField(
        null=True,
        max_length=128,
        description="Digest for context reuse.",
    )
    provider: Field[Optional[str]] = CharField(
        null=True,
        max_length=128,
        description="Provider used to build the context.",
    )
    model: Field[Optional[str]] = CharField(
        null=True,
        max_length=128,
        description="Model used to build the context.",
    )
    expires_at: Field[Optional[datetime]] = DatetimeField(
        null=True,
        description="Time when the context expires.",
    )

    class Meta(Record.Meta):
        abstract = False
        table = "contexts"
        table_description = "Reusable context build record."

        indexes = (
            ("tenant_id", "task_id", "created_at"),
            ("tenant_id", "execution_id", "created_at"),
            ("tenant_id", "conversation_id", "created_at"),
        )
