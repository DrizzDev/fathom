from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from pydantic import JsonValue
from tortoise.fields import CharField, DatetimeField, Field, JSONField

from fathom.infrastructure.interaction.orm.models.base import Mutable, Record

if TYPE_CHECKING:
    from datetime import datetime


class RequestRecord(Mutable, Record):
    """
    Persistent model for idempotency request records.
    """

    key: Field[str] = CharField(max_length=255, description="Idempotency key.")
    hash: Field[str] = CharField(max_length=128, description="Request body hash.")

    state: Field[str] = CharField(max_length=64, description="Idempotency lifecycle state.")
    response: Field[Optional[JsonValue]] = JSONField(
        null=True, description="Typed stored response for replay."
    )

    expires_at: Field[datetime] = DatetimeField(description="Time when the request record expires.")

    class Meta(Record.Meta):
        abstract = False
        table = "requests"
        table_description = "Idempotency request state for retry-safe writes."

        indexes = (("tenant_id", "expires_at"),)
        unique_together = (("tenant_id", "key"),)
