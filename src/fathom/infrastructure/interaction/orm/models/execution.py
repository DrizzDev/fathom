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
    TextField,
)

from fathom.infrastructure.interaction.orm.models.base import Mutable, Record

if TYPE_CHECKING:
    from datetime import datetime

    from fathom.infrastructure.interaction.orm.models.conversation import ConversationRecord


class ExecutionRecord(Mutable, Record):
    """
    Persistent model for one user intent execution inside a conversation.
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
        description="Conversation that owns the execution.",
    )
    workflow_id: Field[Optional[str]] = CharField(
        null=True,
        max_length=128,
        description="Runtime workflow correlation identifier.",
    )
    intent: Field[str] = TextField(
        description="User intent or objective that started the execution.",
    )
    state: Field[str] = CharField(
        max_length=64,
        description="Current execution lifecycle state.",
    )
    code: Field[Optional[str]] = CharField(
        null=True,
        max_length=64,
        description="Machine-readable terminal code.",
    )
    detail: Field[str] = TextField(
        null=True,
        description="Human-readable terminal detail.",
    )
    summary: Field[str] = TextField(
        null=True,
        description="Short execution outcome summary.",
    )
    outcome: Field[JsonValue] = JSONField(
        default=dict,
        description="Typed execution outcome payload.",
    )
    started_at: Field[Optional[datetime]] = DatetimeField(
        null=True,
        description="Time when execution work started.",
    )
    completed_at: Field[Optional[datetime]] = DatetimeField(
        null=True,
        description="Time when execution reached a terminal state.",
    )

    class Meta(Record.Meta):
        abstract = False
        table = "executions"
        table_description = "Run-level intent and outcome inside a conversation."

        indexes = (
            ("tenant_id", "state"),
            ("tenant_id", "workflow_id"),
            ("tenant_id", "conversation_id", "created_at"),
        )
