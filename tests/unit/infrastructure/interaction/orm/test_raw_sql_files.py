from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import asyncpg
from tests.unit.infrastructure.interaction.orm.support import PostgresSchema

from fathom.infrastructure.interaction.orm.raw import InteractionSqlFiles, RawSql


class _RawSqlFactory:
    """
    Builds raw SQL executors for bundled interaction SQL files.
    """

    def build(self) -> RawSql:
        """
        Build one raw SQL executor.
        """

        return RawSql(root=InteractionSqlFiles.bundled().root)


class _FixtureWriter:
    """
    Writes minimal rows required by raw SQL smoke tests.
    """

    def __init__(self, *, connection: asyncpg.Connection) -> None:
        """
        Capture the target connection.
        """

        self.__connection = connection
        self.tenant = "tenant-a"
        self.actor = str(uuid4())
        self.thread = str(uuid4())
        self.execution = str(uuid4())

    async def create_thread(self) -> None:
        """
        Insert an actor and thread pair.
        """

        now = datetime.now(tz=timezone.utc)
        await self.__connection.execute(
            """
            INSERT INTO actors (
                id, tenant_id, kind, name, skills, metadata, created_at, updated_at
            )
            VALUES ($1, $2, 'human', 'Operator', '[]'::jsonb, '{}'::jsonb, $3, $3)
            """,
            self.actor,
            self.tenant,
            now,
        )
        await self.__connection.execute(
            """
            INSERT INTO conversations (
                id, tenant_id, created_by, metadata, created_at, updated_at
            )
            VALUES ($1, $2, $3, '{}'::jsonb, $4, $4)
            """,
            self.thread,
            self.tenant,
            self.actor,
            now,
        )
        await self.__connection.execute(
            """
            INSERT INTO executions (
                id, tenant_id, conversation_id, intent, state, outcome,
                started_at, metadata, created_at, updated_at
            )
            VALUES (
                $1, $2, $3, 'Run smoke', 'running', '{}'::jsonb,
                $4, '{}'::jsonb, $4, $4
            )
            """,
            self.execution,
            self.tenant,
            self.thread,
            now,
        )

    async def create_job(self, *, identifier: str, available_at: datetime) -> None:
        """
        Insert one pending execution job.
        """

        now = datetime.now(tz=timezone.utc)
        await self.__connection.execute(
            """
            INSERT INTO jobs (
                id, tenant_id, conversation_id, execution_id, kind, state, attempts, available_at,
                payload, metadata, created_at, updated_at
            )
            VALUES (
                $1, $2, $3, $4, 'execution', 'pending', 0, $5,
                '{}'::jsonb, '{}'::jsonb, $6, $6
            )
            """,
            identifier,
            self.tenant,
            self.thread,
            self.execution,
            available_at,
            now,
        )

    async def create_message(self, *, identifier: str, text: str) -> None:
        """
        Insert one message row that populates the search table.
        """

        await self.__connection.execute(
            """
            INSERT INTO messages (
                id, tenant_id, conversation_id, execution_id, author, sequence, kind, audience, body,
                labels, metadata, created_at, updated_at
            )
            VALUES (
                $1, $2, $3, $4, $5, 1, 'request', '["thread"]'::jsonb,
                jsonb_build_object('text', $6::text), '[]'::jsonb, '{}'::jsonb, $7, $7
            )
            """,
            identifier,
            self.tenant,
            self.thread,
            self.execution,
            self.actor,
            text,
            datetime.now(tz=timezone.utc),
        )

    async def create_event(self, *, sequence: int) -> None:
        """
        Insert one lifecycle event row for the fixture thread.
        """

        await self.__connection.execute(
            """
            INSERT INTO events (
                id, tenant_id, conversation_id, sequence, kind, source,
                payload, metadata, created_at
            )
            VALUES (
                $1, $2, $3, $4, 'thread.created', 'interaction',
                '{}'::jsonb, '{}'::jsonb, $5
            )
            """,
            str(uuid4()),
            self.tenant,
            self.thread,
            sequence,
            datetime.now(tz=timezone.utc),
        )


class TestRawSqlFiles:
    """
    Verify raw SQL files against local Postgres.
    """

    async def test_claim_job_claims_one_pending_job(self) -> None:
        """
        Claim the oldest available pending job.
        """

        async with PostgresSchema(prefix="conversation_raw_sql") as connection:
            fixture = _FixtureWriter(connection=connection)
            await fixture.create_thread()
            first = str(uuid4())
            second = str(uuid4())
            now = datetime.now(tz=timezone.utc)
            await fixture.create_job(identifier=first, available_at=now)
            await fixture.create_job(
                identifier=second,
                available_at=now + timedelta(seconds=1),
            )
            raw = _RawSqlFactory().build()

            row = await raw.fetchrow(
                connection=connection,
                name="jobs/claim.sql",
                tenant=fixture.tenant,
                owner="worker-a",
                locked_at=now,
                available_at=now,
                job=None,
                kind="execution",
            )

            assert row is not None
            assert row["id"] == first
            assert row["state"] == "claimed"
            assert row["owner"] == "worker-a"

    async def test_claim_job_honors_specific_job_filter(self) -> None:
        """
        Verify the raw claim query can target one pending job by id.
        """

        async with PostgresSchema(prefix="conversation_raw_sql") as connection:
            fixture = _FixtureWriter(connection=connection)
            await fixture.create_thread()
            first = str(uuid4())
            second = str(uuid4())
            now = datetime.now(tz=timezone.utc)
            await fixture.create_job(identifier=first, available_at=now)
            await fixture.create_job(identifier=second, available_at=now)
            raw = _RawSqlFactory().build()

            row = await raw.fetchrow(
                connection=connection,
                name="jobs/claim.sql",
                tenant=fixture.tenant,
                owner="worker-a",
                locked_at=now,
                available_at=now,
                job=second,
                kind="execution",
            )

            assert row is not None
            assert row["id"] == second

    async def test_allocate_sequence_is_concurrent_safe(self) -> None:
        """
        Allocate unique values under concurrent workers.
        """

        schema = PostgresSchema(prefix="conversation_raw_sql")
        async with schema as connection:
            fixture = _FixtureWriter(connection=connection)
            await fixture.create_thread()
            raw = _RawSqlFactory().build()

            async def allocate() -> int:
                worker = await asyncpg.connect(database="postgres")
                try:
                    await worker.execute(f"SET search_path TO {schema.name}")
                    row = await raw.fetchrow(
                        connection=worker,
                        name="sequences/allocate.sql",
                        id=str(uuid4()),
                        tenant=fixture.tenant,
                        thread=fixture.thread,
                        scope="message",
                    )
                    assert row is not None
                    value = row["value"]
                    assert isinstance(value, int)
                    return value
                finally:
                    await worker.close()

            values = await asyncio.gather(*(allocate() for _ in range(5)))

            assert sorted(values) == [1, 2, 3, 4, 5]

    async def test_search_messages_returns_matching_message(self) -> None:
        """
        Search indexed message text through the bundled query.
        """

        async with PostgresSchema(prefix="conversation_raw_sql") as connection:
            fixture = _FixtureWriter(connection=connection)
            await fixture.create_thread()
            message = str(uuid4())
            await fixture.create_message(
                identifier=message,
                text="Find red running shoes",
            )
            raw = _RawSqlFactory().build()

            rows = await raw.fetch(
                connection=connection,
                name="search/messages.sql",
                tenant=fixture.tenant,
                thread=fixture.thread,
                query="running",
                limit=10,
            )

            assert len(rows) == 1
            assert rows[0]["source_id"] == message
            assert rows[0]["execution_id"] == fixture.execution

    async def test_touch_conversation_updates_when_event_is_current(self) -> None:
        """
        Update conversation digest when no newer event exists.
        """

        async with PostgresSchema(prefix="conversation_raw_sql") as connection:
            fixture = _FixtureWriter(connection=connection)
            await fixture.create_thread()
            updated = datetime.now(tz=timezone.utc) + timedelta(seconds=1)
            raw = _RawSqlFactory().build()

            result = await raw.execute(
                connection=connection,
                name="conversations/touch.sql",
                tenant=fixture.tenant,
                thread=fixture.thread,
                sequence=1,
                updated=updated,
                digest="digest",
            )
            row = await raw.fetchrow(
                connection=connection,
                name="conversations/exists.sql",
                tenant=fixture.tenant,
                thread=fixture.thread,
            )

            assert result == "UPDATE 1"
            assert row is not None

    async def test_touch_conversation_skips_stale_event(self) -> None:
        """
        Ignore a touch request when a newer lifecycle event exists.
        """

        async with PostgresSchema(prefix="conversation_raw_sql") as connection:
            fixture = _FixtureWriter(connection=connection)
            await fixture.create_thread()
            await fixture.create_event(sequence=3)
            updated = datetime.now(tz=timezone.utc) + timedelta(seconds=1)
            raw = _RawSqlFactory().build()

            result = await raw.execute(
                connection=connection,
                name="conversations/touch.sql",
                tenant=fixture.tenant,
                thread=fixture.thread,
                sequence=1,
                updated=updated,
                digest="digest",
            )

            assert result == "UPDATE 0"
