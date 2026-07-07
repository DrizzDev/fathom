from __future__ import annotations

from typing import List, Optional, Type
from uuid import uuid4

import asyncpg
import pytest

from fathom.infrastructure.interaction.orm.migration import PostgresMigrator

pytestmark = pytest.mark.asyncio


class MigrationSchemaHarness:
    """
    Owns one disposable local Postgres schema for migration-path tests.
    """

    def __init__(self) -> None:
        """
        Build an isolated schema name.
        """

        self.name = f"live_pkmig_{uuid4().hex}"
        self.connection: Optional[asyncpg.Connection] = None

    async def __aenter__(self) -> MigrationSchemaHarness:
        """
        Create the disposable schema or skip when Postgres is unavailable.
        """

        try:
            self.connection = await asyncpg.connect(database="postgres")
        except (OSError, asyncpg.PostgresError) as exception:
            pytest.skip(f"Local Postgres unavailable: {exception}")

        await self.connection.execute(f'CREATE SCHEMA "{self.name}"')
        await self.connection.execute(f'SET search_path TO "{self.name}"')
        return self

    async def __aexit__(
        self,
        exception_type: Optional[Type[BaseException]],
        exception: Optional[BaseException],
        traceback: Optional[object],
    ) -> None:
        """
        Drop the disposable schema and close the connection.
        """

        _ = exception_type, exception, traceback
        if self.connection is None:
            return

        await self.connection.execute(f'DROP SCHEMA IF EXISTS "{self.name}" CASCADE')
        await self.connection.close()
        self.connection = None

    async def apply(self) -> None:
        """
        Run every pending migration against the schema.
        """

        assert self.connection is not None
        await self.connection.execute(f'SET search_path TO "{self.name}"')
        await PostgresMigrator().apply(connection=self.connection)

    async def primary_key(self, *, table: str) -> str:
        """
        Return the primary-key definition for one table.
        """

        assert self.connection is not None
        definition = await self.connection.fetchval(
            "SELECT pg_get_constraintdef(constraint_record.oid) "
            "FROM pg_constraint constraint_record "
            "JOIN pg_class table_record ON table_record.oid = constraint_record.conrelid "
            "JOIN pg_namespace namespace_record "
            "  ON namespace_record.oid = table_record.relnamespace "
            "WHERE namespace_record.nspname = $1 "
            "  AND table_record.relname = $2 "
            "  AND constraint_record.contype = 'p'",
            self.name,
            table,
        )
        return definition or ""

    async def applied_versions(self) -> List[int]:
        """
        Return every recorded migration version in order.
        """

        assert self.connection is not None
        rows = await self.connection.fetch("SELECT version FROM migrations ORDER BY version")
        return [row["version"] for row in rows]

    async def revert_to_id_only_key(self, *, table: str) -> None:
        """
        Rewrite one table back to the legacy id-only primary key.
        """

        assert self.connection is not None
        await self.connection.execute(f'ALTER TABLE "{table}" DROP CONSTRAINT "{table}_pkey"')
        await self.connection.execute(f'ALTER TABLE "{table}" ADD PRIMARY KEY (id)')

    async def forget_migration(self, *, version: int) -> None:
        """
        Remove one recorded migration version to force re-application.
        """

        assert self.connection is not None
        await self.connection.execute("DELETE FROM migrations WHERE version = $1", version)

    async def seed_actor(self, *, tenant: str, identity: str) -> None:
        """
        Insert one minimal actor row for preservation checks.
        """

        assert self.connection is not None
        await self.connection.execute(
            "INSERT INTO actors (id, tenant_id, kind, name) VALUES ($1, $2, $3, $4)",
            identity,
            tenant,
            "agent",
            identity,
        )

    async def actor_exists(self, *, tenant: str, identity: str) -> bool:
        """
        Return whether one actor row is present.
        """

        assert self.connection is not None
        row = await self.connection.fetchrow(
            "SELECT 1 FROM actors WHERE id = $1 AND tenant_id = $2", identity, tenant
        )
        return row is not None


class TestCompositeKeyMigrationLive:
    """
    Exercise the tenant-scoped primary-key migration against real Postgres.
    """

    async def test_fresh_apply_produces_composite_keys(self) -> None:
        """
        A fresh migration run keys actors and policies by tenant.
        """

        async with MigrationSchemaHarness() as harness:
            await harness.apply()

            assert await harness.applied_versions() == [1, 2]
            assert await harness.primary_key(table="actors") == "PRIMARY KEY (tenant_id, id)"
            assert await harness.primary_key(table="policies") == "PRIMARY KEY (tenant_id, id)"

    async def test_reapply_is_idempotent(self) -> None:
        """
        Re-running the migrator changes nothing and never errors.
        """

        async with MigrationSchemaHarness() as harness:
            await harness.apply()
            await harness.apply()

            assert await harness.applied_versions() == [1, 2]
            assert await harness.primary_key(table="actors") == "PRIMARY KEY (tenant_id, id)"

    async def test_legacy_id_only_key_is_promoted_and_rows_preserved(self) -> None:
        """
        A legacy id-only schema is promoted to a composite key without losing rows.
        """

        async with MigrationSchemaHarness() as harness:
            await harness.apply()
            await harness.revert_to_id_only_key(table="actors")
            await harness.revert_to_id_only_key(table="policies")
            await harness.forget_migration(version=2)
            await harness.seed_actor(tenant="343", identity="agent:fathom")

            assert await harness.primary_key(table="actors") == "PRIMARY KEY (id)"

            await harness.apply()

            assert await harness.applied_versions() == [1, 2]
            assert await harness.primary_key(table="actors") == "PRIMARY KEY (tenant_id, id)"
            assert await harness.primary_key(table="policies") == "PRIMARY KEY (tenant_id, id)"
            assert await harness.actor_exists(tenant="343", identity="agent:fathom") is True
