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
)

from fathom.infrastructure.interaction.orm.models.base import Mutable, Record

if TYPE_CHECKING:
    from datetime import datetime

    from fathom.infrastructure.interaction.orm.models.conversation import ConversationRecord
    from fathom.infrastructure.interaction.orm.models.execution import ExecutionRecord


class JobRecord(Mutable, Record):
    """
    Persistent model for durable background jobs.
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
        description="Conversation that owns the job.",
    )
    execution: ForeignKeyRelation[ExecutionRecord] = ForeignKeyField(
        related_name=False,
        db_constraint=False,
        on_delete=NO_ACTION,
        to="models.ExecutionRecord",
        source_field="execution_id",
        description="Execution that owns the job.",
    )
    task_id: Field[Optional[str]] = CharField(
        null=True,
        max_length=36,
        description="Optional task identifier for the job.",
    )

    kind: Field[str] = CharField(max_length=64, description="Job category.")
    state: Field[str] = CharField(max_length=64, description="Job lifecycle state.")

    attempts: Field[int] = IntField(description="Non-negative job attempt count.")
    owner: Field[Optional[str]] = CharField(
        null=True,
        max_length=128,
        description="Worker currently owning the job.",
    )

    locked_at: Field[Optional[datetime]] = DatetimeField(
        null=True,
        description="Time when a worker claimed the job.",
    )
    available_at: Field[datetime] = DatetimeField(
        description="Earliest time when the job may be claimed.",
    )

    payload: Field[JsonValue] = JSONField(default=dict, description="Typed job payload.")
    code: Field[Optional[str]] = CharField(
        null=True,
        max_length=64,
        description="Machine-readable job failure code.",
    )
    detail: Field[Optional[str]] = CharField(
        null=True,
        max_length=4096,
        description="Human-readable job failure detail.",
    )

    class Meta(Record.Meta):
        abstract = False
        table = "jobs"
        table_description = "Durable background job for conversation maintenance."

        indexes = (
            ("tenant_id", "execution_id"),
            ("tenant_id", "conversation_id"),
            ("tenant_id", "state", "available_at", "kind"),
        )
