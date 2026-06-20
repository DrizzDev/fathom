from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from contextvars import ContextVar, Token
from logging import getLogger
from typing import Any, AsyncGenerator, Optional

import aiosqlite

from fathom.constants.storage import (
    INTERACTION_SQLITE_FOREIGN_KEYS,
    INTERACTION_SQLITE_TEMP_STORE,
)
from fathom.core.exceptions import InteractionError
from fathom.infrastructure.interaction.pypika.sqlite.migration import Migration
from fathom.infrastructure.interaction.timing import SlowQueryLogger, TimedConnection
from fathom.schemas.configuration import (
    SQLiteInteractionConfiguration,  # noqa: TC001 — runtime type for __init__
)


class Unit:
    """
    Transaction and schema owner for SQLite interaction storage.
    """

    def __init__(self, *, configuration: SQLiteInteractionConfiguration) -> None:
        """
        Initialize the unit from a typed SQLite configuration.
        """

        self.__configuration = configuration
        configuration.path.parent.mkdir(parents=True, exist_ok=True)

        self.__initialized = False
        self.__lock = asyncio.Lock()
        self.__connection: ContextVar[Optional[Any]] = ContextVar(
            "sqlite_interaction_connection",
            default=None,
        )
        self.__slow_query_logger = SlowQueryLogger(
            logger=getLogger("fathom.interaction.sqlite"),
            threshold_milliseconds=configuration.slow_query_threshold,
            backend="sqlite",
        )

    async def initialize(self) -> None:
        """
        Create or migrate the interaction schema, applying PRAGMAs from config.
        """

        if self.__initialized:
            return

        async with self.__lock:
            if self.__initialized:
                return

            async with aiosqlite.connect(self.__configuration.path) as connection:
                await self.__apply_persistent_pragmas(connection=connection)
                await self.__apply_connection_pragmas(connection=connection)

                await Migration.migrate(connection=connection)

                await connection.commit()

            self.__initialized = True

    @asynccontextmanager
    async def session(self) -> AsyncGenerator[Any, None]:
        """
        Yield one SQLite connection inside a managed transaction.
        """

        existing = self.__connection.get()

        if existing is not None:
            yield existing
            return

        await self.initialize()
        connection = await aiosqlite.connect(self.__configuration.path)

        connection.row_factory = aiosqlite.Row
        wrapped: Any = (
            TimedConnection(inner=connection, logger=self.__slow_query_logger)
            if self.__slow_query_logger.threshold > 0
            else connection
        )
        token: Token[Optional[Any]] = self.__connection.set(wrapped)

        try:
            await self.__apply_connection_pragmas(connection=connection)
            await connection.execute("BEGIN IMMEDIATE")

            yield wrapped
            await connection.commit()
        except aiosqlite.IntegrityError as exception:
            await connection.rollback()
            raise InteractionError(
                f"Interaction storage integrity violation: {exception}"
            ) from exception
        except Exception:
            await connection.rollback()
            raise
        finally:
            self.__connection.reset(token)
            await connection.close()

    @asynccontextmanager
    async def atomic(self) -> AsyncGenerator[None, None]:
        """
        Open one grouped transaction boundary that reuses the active session.
        """

        async with self.session():
            yield

    async def __apply_persistent_pragmas(self, *, connection: aiosqlite.Connection) -> None:
        """
        Apply file-level PRAGMAs that persist in the database header.

        Values come from the validated SQLiteJournalMode/SQLiteSynchronous
        enums, so f-string interpolation is bounded to the enum's value set.
        """

        configuration = self.__configuration

        await connection.execute(f"PRAGMA synchronous = {configuration.synchronous.value}")
        await connection.execute(f"PRAGMA journal_mode = {configuration.journal_mode.value}")

    async def __apply_connection_pragmas(self, *, connection: aiosqlite.Connection) -> None:
        """
        Apply connection-scoped PRAGMAs that must run on every new connection.
        """

        configuration = self.__configuration

        await connection.execute(f"PRAGMA mmap_size = {configuration.mmap_size}")
        await connection.execute(f"PRAGMA busy_timeout = {configuration.busy_timeout}")
        await connection.execute(f"PRAGMA temp_store = {INTERACTION_SQLITE_TEMP_STORE}")

        if INTERACTION_SQLITE_FOREIGN_KEYS:
            await connection.execute("PRAGMA foreign_keys = ON")
