from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from pydantic import JsonValue
from tortoise.fields import CharField, DatetimeField, Field, JSONField
from tortoise.models import Model

if TYPE_CHECKING:
    from datetime import datetime


class Record(Model):
    """
    Base row for tenant-scoped persisted conversation data.
    """

    id: Field[str] = CharField(
        max_length=36,
        primary_key=True,
        description="Opaque row identifier.",
    )
    tenant_id: Field[str] = CharField(
        db_index=True,
        max_length=128,
        description="Tenant that owns the row.",
    )
    workspace_id: Field[Optional[str]] = CharField(
        null=True,
        max_length=128,
        description="Optional workspace scope inside the tenant.",
    )

    metadata: Field[JsonValue] = JSONField(
        default=dict,
        description="Typed extension metadata stored as JSON.",
    )

    created_at: Field[datetime] = DatetimeField(
        auto_now_add=True,
        description="Time when the row was created.",
    )
    created_by: Field[Optional[str]] = CharField(
        null=True,
        max_length=128,
        description="Actor or operator that created the row.",
    )

    class Meta(Model.Meta):
        abstract = True


class Mutable:
    """
    Adds mutable row timestamps and soft-delete attribution.
    """

    updated_at: Field[datetime] = DatetimeField(
        auto_now=True,
        description="Time when the row was last modified.",
    )
    updated_by: Field[Optional[str]] = CharField(
        null=True,
        max_length=128,
        description="Actor or operator that last modified the row.",
    )

    deleted_at: Field[Optional[datetime]] = DatetimeField(
        null=True,
        description="Time when the row was soft-deleted.",
    )
    deleted_by: Field[Optional[str]] = CharField(
        null=True,
        max_length=128,
        description="Actor or operator that soft-deleted the row.",
    )


__all__ = ["Mutable", "Record"]
