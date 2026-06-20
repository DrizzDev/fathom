"""
Shared fixtures for Postgres interaction-adapter integration tests.

These tests are gated on the `FATHOM_TEST_POSTGRES_DSN` environment variable.
When the variable is unset, every test in this directory is skipped so the
integration suite is opt-in: developers without a local Postgres do not pay
the connection cost, and CI pipelines explicitly select Postgres-bearing
runners by setting the variable. The DSN must point at a database the test
process can `CREATE SCHEMA` and `DROP SCHEMA` on.

Each test gets its own ephemeral schema (`fathom_test_<uuid>`) so:
  - parallel test runners do not collide on table state
  - a failing test cannot poison subsequent runs
  - teardown is a single `DROP SCHEMA ... CASCADE`

The `postgres_adapter` fixture wires a `PostgresInteraction` against the
ephemeral schema and runs the full migration bundle (TABLES, INDEXES,
BACKFILLS, MIGRATIONS, VIEWS, GRANTS) inside that schema, exercising the
real declarative bundle the production deployment runs.
"""

from __future__ import annotations

import os
import uuid
from typing import TYPE_CHECKING

import pytest
import pytest_asyncio

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


_DSN = os.environ.get("FATHOM_TEST_POSTGRES_DSN")

requires_postgres = pytest.mark.skipif(
    _DSN is None,
    reason=(
        "FATHOM_TEST_POSTGRES_DSN is not set; Postgres integration tests are "
        "skipped. Set the variable to a DSN that asyncpg accepts to enable."
    ),
)


@pytest_asyncio.fixture()
async def postgres_dsn() -> str:
    """
    Return the Postgres DSN, failing fast if the env was not set.
    """

    if _DSN is None:
        pytest.skip("FATHOM_TEST_POSTGRES_DSN not set")
    return _DSN


@pytest_asyncio.fixture()
async def postgres_schema(postgres_dsn: str) -> AsyncIterator[str]:
    """
    Provision an ephemeral Postgres schema for one test and drop it after.

    The schema name is a fresh UUID-suffixed identifier so concurrent test
    runners cannot collide. The fixture skips at runtime if asyncpg is not
    installed (e.g. on dev workstations without the optional dependency).
    """

    asyncpg = pytest.importorskip("asyncpg")
    schema_name = f"fathom_test_{uuid.uuid4().hex[:12]}"
    connection = await asyncpg.connect(dsn=postgres_dsn)
    try:
        await connection.execute(f'CREATE SCHEMA "{schema_name}"')
    except Exception:
        await connection.close()
        raise

    try:
        yield schema_name
    finally:
        try:
            await connection.execute(f'DROP SCHEMA "{schema_name}" CASCADE')
        finally:
            await connection.close()


@pytest_asyncio.fixture()
async def postgres_adapter(postgres_dsn: str, postgres_schema: str):
    """
    Build a PostgresInteraction adapter pinned to the ephemeral schema.

    The adapter runs the full migration bundle on first use, so individual
    tests can exercise the public InteractionPort surface without manual
    schema setup. `aclose()` is called on teardown so the asyncpg pool is
    drained between tests.
    """

    from fathom.adapters.interaction.pypika.postgres import PostgresInteraction
    from fathom.schemas.configuration import PostgresInteractionConfiguration

    configuration = PostgresInteractionConfiguration(
        dsn=postgres_dsn,
        schema_name=postgres_schema,
    )
    adapter = PostgresInteraction(configuration=configuration)
    try:
        yield adapter
    finally:
        await adapter.aclose()
