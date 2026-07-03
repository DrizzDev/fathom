from __future__ import annotations

from uuid import uuid4

import asyncpg
import pytest

from fathom.adapters.interaction.orm.postgres import PostgresInteraction
from fathom.constants.storage import PostgresMigrationMode
from fathom.core.exceptions import StorageConfigurationError
from fathom.infrastructure.interaction.orm.migration import PostgresSchemaValidationError
from fathom.schemas.configuration import PostgresInteractionConfiguration


class TestPostgresInteractionAdapter:
    """
    Verify lifecycle behavior of the ORM-backed Postgres adapter.
    """

    async def test_initialize_applies_migration_and_close_is_idempotent(self) -> None:
        """
        Initialize a disposable schema and close ORM connections safely.
        """

        schema = f"interaction_adapter_{uuid4().hex}"
        adapter = PostgresInteraction(
            configuration=PostgresInteractionConfiguration(
                dsn="postgresql://localhost/postgres",
                schema_name=schema,
                pool_min_size=1,
                pool_max_size=2,
                migration_mode=PostgresMigrationMode.APPLY,
            )
        )
        connection = await self.__connect()
        try:
            await adapter.initialize()
            await adapter.initialize()

            row = await connection.fetchrow(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = $1 AND table_name = 'conversations'
                """,
                schema,
            )
            assert row is not None
            stale = await connection.fetchrow(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = $1 AND table_name = 'threads'
                """,
                schema,
            )
            assert stale is None
        finally:
            await adapter.aclose()
            await adapter.aclose()
            await connection.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
            await connection.close()

    async def test_initialize_validates_existing_schema_without_mutating_migrations(
        self,
    ) -> None:
        """
        Validate mode starts against an existing schema without rewriting migration metadata.
        """

        schema = f"interaction_validate_{uuid4().hex}"
        apply_adapter = PostgresInteraction(
            configuration=PostgresInteractionConfiguration(
                dsn="postgresql://localhost/postgres",
                schema_name=schema,
                pool_min_size=1,
                pool_max_size=2,
                migration_mode=PostgresMigrationMode.APPLY,
            )
        )
        validate_adapter = PostgresInteraction(
            configuration=PostgresInteractionConfiguration(
                dsn="postgresql://localhost/postgres",
                schema_name=schema,
                pool_min_size=1,
                pool_max_size=2,
                migration_mode=PostgresMigrationMode.VALIDATE,
            )
        )
        connection = await self.__connect()
        try:
            await apply_adapter.initialize()
            before = await self.__migration_count(connection=connection, schema=schema)

            await apply_adapter.aclose()
            await validate_adapter.initialize()
            after = await self.__migration_count(connection=connection, schema=schema)

            assert after == before
        finally:
            await validate_adapter.aclose()
            await apply_adapter.aclose()
            await connection.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
            await connection.close()

    async def test_initialize_validate_mode_does_not_create_missing_schema(self) -> None:
        """
        Validate mode fails fast and leaves a missing schema missing.
        """

        schema = f"interaction_missing_{uuid4().hex}"
        adapter = PostgresInteraction(
            configuration=PostgresInteractionConfiguration(
                dsn="postgresql://localhost/postgres",
                schema_name=schema,
                pool_min_size=1,
                pool_max_size=2,
                migration_mode=PostgresMigrationMode.VALIDATE,
            )
        )
        connection = await self.__connect()
        try:
            with pytest.raises(PostgresSchemaValidationError, match="configured schema"):
                await adapter.initialize()

            row = await connection.fetchrow(
                """
                SELECT schema_name
                FROM information_schema.schemata
                WHERE schema_name = $1
                """,
                schema,
            )
            assert row is None
        finally:
            await adapter.aclose()
            await connection.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
            await connection.close()

    async def test_initialize_rejects_sqlalchemy_style_dsn_scheme(self) -> None:
        """
        Reject `postgresql+asyncpg` DSNs before opening a network connection.
        """

        adapter = PostgresInteraction(
            configuration=PostgresInteractionConfiguration(
                dsn="postgresql+asyncpg://user:secret@localhost/postgres",
            )
        )

        with pytest.raises(StorageConfigurationError, match="DSN scheme"):
            await adapter.initialize()

    async def __connect(self) -> asyncpg.Connection:
        """
        Open a local Postgres connection or skip when Postgres is unavailable.
        """

        try:
            return await asyncpg.connect(database="postgres")
        except (OSError, asyncpg.PostgresError) as exception:
            pytest.skip(f"Local Postgres unavailable: {exception}")

    async def __migration_count(
        self,
        *,
        connection: asyncpg.Connection,
        schema: str,
    ) -> int:
        """
        Count migration rows recorded in a disposable schema.
        """

        row = await connection.fetchrow(f'SELECT COUNT(*) AS total FROM "{schema}".migrations')
        assert row is not None
        return int(row["total"])
