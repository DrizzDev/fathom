from __future__ import annotations

from typing import Awaitable, Dict, Protocol

import aiosqlite

from fathom.infrastructure.interaction.pypika.sqlite import schema


class Step(Protocol):
    """
    Callable migration step that accepts a SQLite connection.
    """

    def __call__(self, *, connection: aiosqlite.Connection) -> Awaitable[None]:
        """
        Execute one schema migration step.
        """

        ...


class Migration:
    """
    SQLite schema migrator for the first public Fathom conversation baseline.
    """

    CURRENT = 1

    @classmethod
    async def migrate(cls, *, connection: aiosqlite.Connection) -> None:
        """
        Upgrade an empty or older local database to the current baseline.
        """

        current = await cls.__current(connection=connection)
        steps = cls.steps()

        for source in range(current, cls.CURRENT):
            step = steps.get(source)
            if step is None:
                raise RuntimeError(f"Missing interaction migration step {source} -> {source + 1}")

            await step(connection=connection)
            await cls.__write(connection=connection, version=source + 1)

    @classmethod
    def steps(cls) -> Dict[int, Step]:
        """
        Return migration steps keyed by source version.
        """

        return {0: cls.__create_baseline}

    @classmethod
    async def __create_baseline(cls, *, connection: aiosqlite.Connection) -> None:
        """
        Create the complete Fathom conversation schema baseline.
        """

        for statement in (
            schema.ACTOR,
            schema.THREAD,
            schema.MEMBERSHIP,
            schema.TASK,
            schema.MESSAGE,
            schema.EVENT,
            schema.ARTIFACT,
            schema.SCRIPT,
            schema.SCRIPT_VERSION,
            schema.POLICY,
            schema.JOB,
            schema.REQUESTS,
            schema.CONTEXT,
            schema.SEQUENCES,
        ):
            await connection.execute(statement)

        if not await cls.__column_exists(
            connection=connection, table="messages", column="body_text"
        ):
            await connection.execute(schema.MESSAGES_BODY_TEXT_COLUMN)

        for statement in (*schema.INDEXES, *schema.ACTIVE_INDEXES):
            await connection.execute(statement)

        await connection.execute(schema.SEARCH_TABLE)
        await connection.execute(schema.SEARCH_TRIGGER_INSERT)
        await connection.execute(schema.SEARCH_TRIGGER_DELETE)
        await connection.execute(schema.SEARCH_TRIGGER_UPDATE)
        await connection.execute(schema.SEARCH_BACKFILL)

    @classmethod
    async def __column_exists(
        cls, *, connection: aiosqlite.Connection, table: str, column: str
    ) -> bool:
        """
        Return True when the named column already exists on the given table.
        """

        async with connection.execute(f"PRAGMA table_xinfo({table})") as cursor:
            rows = await cursor.fetchall()

        return any(str(row[1]) == column for row in rows)

    @staticmethod
    async def __current(*, connection: aiosqlite.Connection) -> int:
        """
        Read the stored schema version via PRAGMA user_version.
        """

        async with connection.execute("PRAGMA user_version") as cursor:
            row = await cursor.fetchone()

        if row is None:
            return 0

        return int(row[0])

    @staticmethod
    async def __write(*, connection: aiosqlite.Connection, version: int) -> None:
        """
        Persist the latest schema version via PRAGMA user_version.
        """

        if not isinstance(version, int) or version < 0:
            raise ValueError(f"Schema version must be a non-negative int, got {version!r}.")

        await connection.execute(f"PRAGMA user_version = {int(version)}")  # nosec B608
