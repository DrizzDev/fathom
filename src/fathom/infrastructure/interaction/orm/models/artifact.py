from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from pydantic import JsonValue
from tortoise.fields import (
    NO_ACTION,
    BigIntField,
    CharField,
    Field,
    ForeignKeyField,
    ForeignKeyRelation,
    JSONField,
    TextField,
)

from fathom.infrastructure.interaction.orm.models.base import Mutable, Record

if TYPE_CHECKING:
    from fathom.infrastructure.interaction.orm.models.conversation import ConversationRecord
    from fathom.infrastructure.interaction.orm.models.execution import ExecutionRecord


class ArtifactRecord(Mutable, Record):
    """
    Persistent model for artifact references produced by executions.
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
        description="Conversation that owns the artifact.",
    )
    execution: ForeignKeyRelation[ExecutionRecord] = ForeignKeyField(
        related_name=False,
        db_constraint=False,
        on_delete=NO_ACTION,
        to="models.ExecutionRecord",
        source_field="execution_id",
        description="Execution that owns the artifact.",
    )

    task_id: Field[Optional[str]] = CharField(
        null=True,
        max_length=36,
        description="Optional task identifier that produced the artifact.",
    )
    producer: Field[Optional[str]] = CharField(
        null=True,
        max_length=128,
        description="Actor that produced the artifact.",
    )

    uri: Field[str] = TextField(description="Permanent artifact location.")
    kind: Field[str] = CharField(max_length=64, description="Artifact category.")
    backend: Field[str] = CharField(max_length=64, description="Artifact storage backend.")

    mime: Field[Optional[str]] = CharField(
        null=True,
        max_length=255,
        description="Artifact media type.",
    )
    size: Field[Optional[int]] = BigIntField(
        null=True,
        description="Non-negative artifact size.",
    )

    retention: Field[Optional[str]] = CharField(
        null=True,
        max_length=64,
        description="Optional artifact retention class.",
    )
    labels: Field[JsonValue] = JSONField(
        default=list,
        description="Typed labels attached to the artifact.",
    )

    class Meta(Record.Meta):
        abstract = False
        table = "artifacts"
        table_description = "Stored artifact reference without transient signing state."

        indexes = (
            ("tenant_id", "task_id"),
            ("tenant_id", "deleted_at"),
            ("tenant_id", "execution_id"),
            ("tenant_id", "conversation_id"),
        )
