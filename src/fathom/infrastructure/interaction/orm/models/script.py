from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from pydantic import JsonValue
from tortoise.fields import (
    NO_ACTION,
    CharField,
    Field,
    ForeignKeyField,
    ForeignKeyRelation,
    IntField,
    JSONField,
    TextField,
)

from fathom.infrastructure.interaction.orm.models.base import Mutable, Record

if TYPE_CHECKING:
    from fathom.infrastructure.interaction.orm.models.conversation import ConversationRecord
    from fathom.infrastructure.interaction.orm.models.execution import ExecutionRecord


class ScriptRecord(Mutable, Record):
    """
    Persistent model for the current editable script document.
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
        description="Conversation that owns the script.",
    )
    execution: ForeignKeyRelation[ExecutionRecord] = ForeignKeyField(
        db_index=True,
        related_name=False,
        db_constraint=False,
        on_delete=NO_ACTION,
        to="models.ExecutionRecord",
        source_field="execution_id",
        description="Execution that produced the script.",
    )

    task_id: Field[Optional[str]] = CharField(
        null=True,
        max_length=36,
        description="Optional task identifier that produced the script.",
    )
    title: Field[Optional[str]] = CharField(
        null=True,
        max_length=1024,
        description="Optional script title.",
    )
    format: Field[str] = CharField(
        max_length=64,
        description="Script content format.",
    )
    status: Field[str] = CharField(
        max_length=64,
        description="Current script lifecycle status.",
    )
    content: Field[str] = TextField(
        description="Current script content.",
    )
    revision: Field[int] = IntField(
        description="Positive current script revision.",
    )
    checksum: Field[str] = CharField(
        max_length=128,
        description="Checksum for the current script content.",
    )

    class Meta(Record.Meta):
        abstract = False
        table = "scripts"
        table_description = "Current editable script document for an execution."

        indexes = (
            ("tenant_id", "task_id"),
            ("tenant_id", "deleted_at"),
            ("tenant_id", "execution_id"),
            ("tenant_id", "conversation_id", "updated_at"),
        )


class ScriptVersionRecord(Mutable, Record):
    """
    Persistent model for immutable script content revisions.
    """

    if TYPE_CHECKING:
        script_id: str

    script: ForeignKeyRelation[ScriptRecord] = ForeignKeyField(
        db_index=True,
        related_name=False,
        db_constraint=False,
        on_delete=NO_ACTION,
        to="models.ScriptRecord",
        source_field="script_id",
        description="Script document that owns this revision.",
    )
    version: Field[int] = IntField(
        description="Positive script revision number.",
    )
    source: Field[str] = CharField(
        max_length=64,
        description="Source that created the revision.",
    )
    content: Field[str] = TextField(
        description="Script content captured for the revision.",
    )
    checksum: Field[str] = CharField(
        max_length=128,
        description="Checksum for the revision content.",
    )
    summary: Field[str] = TextField(
        null=True,
        description="Optional human-readable revision summary.",
    )
    actor: Field[Optional[str]] = CharField(
        null=True,
        max_length=128,
        description="Actor that created the revision.",
    )
    metadata: Field[JsonValue] = JSONField(
        default=dict,
        description="Typed extension metadata stored as JSON.",
    )

    class Meta(Record.Meta):
        abstract = False
        table = "script_versions"
        table_description = "Immutable content revision for a script document."

        indexes = (("tenant_id", "script_id", "version"),)
        unique_together = (("tenant_id", "script_id", "version"),)
