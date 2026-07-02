from __future__ import annotations

from typing import Optional

from pydantic import JsonValue
from tortoise.fields import CharField, Field, JSONField, TextField

from fathom.infrastructure.interaction.orm.models.base import Mutable, Record


class ActorRecord(Mutable, Record):
    """
    Persistent model for a conversation actor.
    """

    name: Field[str] = TextField(description="Human-readable actor name.")
    kind: Field[str] = CharField(max_length=64, description="Actor category.")

    external: Field[Optional[str]] = CharField(
        null=True,
        max_length=512,
        description="External identity associated with the actor.",
    )

    runtime: Field[Optional[str]] = CharField(
        null=True,
        max_length=128,
        description="Runtime that owns the actor.",
    )
    provider: Field[Optional[str]] = CharField(
        null=True,
        max_length=128,
        description="Provider backing the actor.",
    )
    model: Field[Optional[str]] = CharField(
        null=True,
        max_length=128,
        description="Model identifier used by the actor.",
    )
    skills: Field[JsonValue] = JSONField(
        default=dict,
        description="Typed actor skill metadata.",
    )

    class Meta(Record.Meta):
        abstract = False
        table = "actors"
        table_description = "Conversation actor scoped by tenant."

        indexes = (("tenant_id", "kind"),)
