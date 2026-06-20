from __future__ import annotations

from typing import Tuple

from pydantic import BaseModel, ConfigDict, Field


class PostgresMigrationStep(BaseModel):
    """
    Defines one append-only Postgres schema migration.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(description="Stable migration name used for diagnostics.")
    version: int = Field(description="Monotonic migration version recorded in the ledger.")
    statements: Tuple[str, ...] = Field(description="SQL statements executed for the migration.")
