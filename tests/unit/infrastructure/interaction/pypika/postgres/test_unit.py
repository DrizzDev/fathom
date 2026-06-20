from __future__ import annotations

from typing import Tuple

import pytest

from fathom.core.exceptions import InteractionError
from fathom.infrastructure.interaction.pypika.postgres import schema
from fathom.infrastructure.interaction.pypika.postgres.unit import PostgresConnection, Unit
from fathom.schemas.configuration import PostgresInteractionConfiguration


class _FakeAsyncpgConnection:
    """
    Minimal asyncpg connection test double for PostgresConnection tests.
    """

    def __init__(self) -> None:
        """
        Initialise captured SQL and parameter fields.
        """

        self.sql: str | None = None
        self.parameters: Tuple[object, ...] = ()

    async def fetch(self, sql: str, *parameters: object):
        """
        Capture fetch SQL and return no rows.
        """

        self.sql = sql
        self.parameters = tuple(parameters)
        return []

    async def execute(self, sql: str, *parameters: object) -> str:
        """
        Capture execute SQL and return a command tag with a rowcount.
        """

        self.sql = sql
        self.parameters = tuple(parameters)
        return "DELETE 2"


@pytest.mark.asyncio
async def test_postgres_connection_passes_sql_through_unchanged():
    """
    The native repositories emit `$n` placeholders directly; the connection
    wrapper is a thin pass-through to asyncpg.
    """

    raw = _FakeAsyncpgConnection()
    connection = PostgresConnection(connection=raw)

    async with connection.execute(
        "SELECT * FROM messages WHERE id = $1 AND task = $2",
        ("message-1", "task-1"),
    ) as cursor:
        assert await cursor.fetchall() == []

    assert raw.sql == "SELECT * FROM messages WHERE id = $1 AND task = $2"
    assert raw.parameters == ("message-1", "task-1")


@pytest.mark.asyncio
async def test_postgres_connection_reports_rowcount_from_command_tag():
    """
    Awaitable execution path returns the rowcount parsed from the asyncpg
    command tag (e.g. `DELETE 2`).
    """

    raw = _FakeAsyncpgConnection()
    connection = PostgresConnection(connection=raw)

    result = await connection.execute(
        "DELETE FROM jobs WHERE tenant = $1",
        ("tenant-1",),
    )

    assert result.rowcount == 2
    assert raw.sql == "DELETE FROM jobs WHERE tenant = $1"
    assert raw.parameters == ("tenant-1",)


def test_postgres_unit_rejects_unsafe_schema_identifier():
    """
    schema_name is interpolated as an identifier and must be validated upfront.
    """

    with pytest.raises(InteractionError):
        Unit(
            configuration=PostgresInteractionConfiguration(
                host="localhost",
                user="fathom",
                password="secret",
                database="fathom",
                schema_name='public"; DROP SCHEMA public; --',
            )
        )


def test_postgres_schema_uses_ordered_append_only_migrations():
    """
    Postgres schema changes must be represented as ordered migration steps,
    not as one mutable declarative bundle.
    """

    versions = [step.version for step in schema.MIGRATION_STEPS]

    assert versions == sorted(versions)
    assert len(versions) == len(set(versions))
    assert versions[-1] == schema.SCHEMA_VERSION
    assert schema.MIGRATION_TABLE not in schema.TABLES


def test_customer_success_views_are_not_managed_by_interaction_schema():
    """
    Customer-success read models and role grants are host-owned concerns, not
    interaction schema bootstrap work.
    """

    migration_sql = "\n".join(
        statement for migration in schema.MIGRATION_STEPS for statement in migration.statements
    )
    bootstrap_sql = "\n".join(schema.BOOTSTRAP_STEPS)

    assert schema.VIEWS == ()
    assert "CREATE OR REPLACE VIEW" not in migration_sql
    assert "readonly_tenants" not in migration_sql
    assert "fathom_readonly" not in bootstrap_sql
    assert "current_user" not in bootstrap_sql


def test_postgres_search_migration_indexes_audit_evidence():
    """
    Search migrations must include audit evidence payloads.
    """

    search_sql = "\n".join(schema.SEARCH_EVIDENCE)

    assert "body->>'evidence'" in search_sql
    assert "UPDATE search AS target" in search_sql


def test_postgres_schema_declares_concrete_foreign_keys():
    """
    Concrete relationship columns must be DB-enforced, not just documented.
    """

    table_sql = "\n".join(schema.TABLES)
    constraint_sql = "\n".join(schema.STRUCTURAL_CONSTRAINTS)

    assert "FOREIGN KEY (tenant, creator) REFERENCES actors(tenant, id)" in table_sql
    assert "FOREIGN KEY (tenant, thread) REFERENCES threads(tenant, id)" in table_sql
    assert "fk_tasks_origin_messages" in constraint_sql
    assert "FOREIGN KEY (tenant, origin) REFERENCES %I.messages(tenant, id)" in constraint_sql


def test_postgres_membership_sql_uses_current_departure_column():
    """
    Membership repository Pypika tables must reference the renamed departed_at
    column rather than the deprecated `left` column.
    """

    from fathom.infrastructure.interaction.pypika.postgres import tables

    memberships = tables.MEMBERSHIPS

    # Pypika Table dynamically attributes column names; verify the canonical
    # current name resolves and that the deprecated alias is not referenced
    # anywhere in the schema by checking the DDL strings.
    departed_field = memberships.departed_at
    assert departed_field.name == "departed_at"

    def _walk(value: object):
        if isinstance(value, str):
            yield value
        elif isinstance(value, (list, tuple, set, frozenset)):
            for item in value:
                yield from _walk(item)
        elif isinstance(value, dict):
            for item in value.values():
                yield from _walk(item)

    joined = "\n".join(_walk(list(schema.__dict__.values())))
    assert "departed_at" in joined
    assert '"left"' not in joined
