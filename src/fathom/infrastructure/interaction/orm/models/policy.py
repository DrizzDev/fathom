from __future__ import annotations

from typing import Optional

from pydantic import JsonValue
from tortoise.fields import CharField, Field, JSONField

from fathom.infrastructure.interaction.orm.models.base import Mutable, Record


class PolicyRecord(Mutable, Record):
    """
    Persistent model for tenant and workspace governance policies.
    """

    name: Field[str] = CharField(max_length=128, description="Policy name.")
    scope: Field[str] = CharField(max_length=64, description="Policy scope.")

    region: Field[Optional[str]] = CharField(
        null=True,
        max_length=128,
        description="Optional region where the policy applies.",
    )
    retention: Field[JsonValue] = JSONField(description="Typed retention policy.")
    labels: Field[JsonValue] = JSONField(default=list, description="Typed label policy.")
    sanitizers: Field[JsonValue] = JSONField(default=list, description="Typed sanitizer policy.")

    memories: Field[JsonValue] = JSONField(default=list, description="Typed memory policy.")
    artifacts: Field[JsonValue] = JSONField(default=list, description="Typed artifact policy.")

    class Meta(Record.Meta):
        abstract = False
        table = "policies"
        table_description = "Governance policy scoped by tenant and workspace."

        indexes = (("tenant_id", "workspace_id", "name"),)
        unique_together = (("tenant_id", "workspace_id", "name"),)
