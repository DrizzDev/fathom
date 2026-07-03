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
    IntField,
    JSONField,
    TextField,
)

from fathom.infrastructure.interaction.orm.models.base import Mutable, Record

if TYPE_CHECKING:
    from datetime import datetime

    from fathom.infrastructure.interaction.orm.models.conversation import ConversationRecord
    from fathom.infrastructure.interaction.orm.models.execution import ExecutionRecord


class TaskRecord(Mutable, Record):
    """
    Persistent model for operational work inside one execution.
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
        description="Conversation that owns the task.",
    )
    execution: ForeignKeyRelation[ExecutionRecord] = ForeignKeyField(
        db_index=True,
        related_name=False,
        db_constraint=False,
        on_delete=NO_ACTION,
        to="models.ExecutionRecord",
        source_field="execution_id",
        description="Execution that owns the task.",
    )

    assignee: Field[Optional[str]] = CharField(
        null=True,
        max_length=128,
        description="Actor assigned to work on the task.",
    )

    parent_id: Field[Optional[str]] = CharField(
        null=True,
        max_length=36,
        description="Optional parent task identifier in the execution tree.",
    )
    root_id: Field[Optional[str]] = CharField(
        null=True,
        max_length=36,
        description="Optional root task identifier in the execution tree.",
    )
    origin_id: Field[Optional[str]] = CharField(
        null=True,
        max_length=36,
        description="Optional origin message identifier that opened the task.",
    )

    objective: Field[str] = TextField(description="Task objective text.")
    kind: Field[str] = CharField(max_length=64, description="Task category.")

    reference: Field[Optional[str]] = CharField(
        null=True, max_length=1024, description="External reference associated with the task."
    )
    state: Field[str] = CharField(max_length=64, description="Current task lifecycle state.")

    code: Field[Optional[str]] = CharField(
        null=True, max_length=64, description="Machine-readable terminal code."
    )
    detail: Field[Optional[str]] = CharField(
        null=True, max_length=4096, description="Human-readable terminal detail."
    )

    summary: Field[str] = TextField(null=True, description="Short task outcome summary.")
    plan: Field[JsonValue] = JSONField(default=dict, description="Typed task plan payload.")
    outcome: Field[JsonValue] = JSONField(default=dict, description="Typed task outcome payload.")
    progress: Field[JsonValue] = JSONField(default=dict, description="Typed task progress payload.")

    started_at: Field[Optional[datetime]] = DatetimeField(
        null=True, description="Time when task work started."
    )
    completed_at: Field[Optional[datetime]] = DatetimeField(
        null=True, description="Time when task reached a terminal state."
    )
    elapsed: Field[Optional[int]] = IntField(
        null=True, description="Non-negative elapsed duration value."
    )

    class Meta(Record.Meta):
        abstract = False
        table = "tasks"
        table_description = "Operational task tree owned by one execution."

        indexes = (
            ("tenant_id", "parent_id"),
            ("tenant_id", "deleted_at"),
            ("tenant_id", "execution_id"),
            ("tenant_id", "conversation_id"),
            ("tenant_id", "conversation_id", "created_at"),
        )
