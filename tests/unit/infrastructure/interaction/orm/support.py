from __future__ import annotations

import asyncio
from contextvars import ContextVar, Token
from typing import TYPE_CHECKING, Optional, Type
from uuid import uuid4

import asyncpg
import pytest

from fathom.infrastructure.interaction.orm.migration import PostgresMigrator
from fathom.infrastructure.interaction.orm.runtime import PostgresInteractionRuntime
from fathom.schemas.configuration import PostgresInteractionConfiguration

if TYPE_CHECKING:
    from contextlib import AbstractAsyncContextManager
    from types import TracebackType

ACTIVE_RUNTIME: ContextVar[Optional[PostgresInteractionRuntime]] = ContextVar(
    "active_interaction_runtime",
    default=None,
)


class LoopDrain:
    """
    Test-only teardown shim; it must never migrate into ``src/``.

    Production connection owners close their pools explicitly and run on a long-lived loop, so
    ``asyncpg``'s ``loop.call_soon`` socket-close callback always fires. This shim only covers the
    ``IsolatedAsyncioTestCase`` teardown race, where the test loop can close before that callback
    runs and the connection is otherwise garbage-collected on a later test's loop. Yielding lets
    the queued callbacks drain on the loop that created the connection.
    """

    __YIELD_ROUNDS: int = 2

    @classmethod
    async def settle(cls) -> None:
        """
        Yield to the running loop enough times to flush queued connection-close callbacks.
        """

        for _ in range(cls.__YIELD_ROUNDS):
            await asyncio.sleep(0)


class InteractionRuntimeRegistry:
    """
    Shares the active repository-test runtime inside one schema context.
    """

    @classmethod
    def require(cls) -> PostgresInteractionRuntime:
        """
        Return the active repository-test runtime.
        """

        runtime = ACTIVE_RUNTIME.get()
        if runtime is None:
            raise RuntimeError("Interaction repository tests require an active schema.")

        return runtime


class PostgresSchema:
    """
    Owns one disposable migrated local Postgres schema.
    """

    def __init__(self, *, prefix: str) -> None:
        """
        Initialize schema state.
        """

        self.name = f"{prefix}_{uuid4().hex}"
        self.connection: Optional[asyncpg.Connection] = None

    async def __aenter__(self) -> asyncpg.Connection:
        """
        Create and migrate the disposable schema.
        """

        try:
            connection = await asyncpg.connect(database="postgres")
        except (OSError, asyncpg.PostgresError) as exception:
            pytest.skip(f"Local Postgres unavailable: {exception}")

        self.connection = connection
        await connection.execute(f"CREATE SCHEMA {self.name}")
        await connection.execute(f"SET search_path TO {self.name}")
        await PostgresMigrator().apply(connection=connection)
        return connection

    async def __aexit__(
        self,
        exception_type: Optional[Type[BaseException]],
        exception: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        """
        Drop the disposable schema and close the connection.
        """

        if self.connection is None:
            return

        await self.connection.execute(f"DROP SCHEMA IF EXISTS {self.name} CASCADE")
        await self.connection.close()
        await LoopDrain.settle()


class InteractionPostgresSchema:
    """
    Owns a disposable migrated schema and a store connection bound to it.
    """

    def __init__(self, *, prefix: str) -> None:
        """
        Initialize schema state.
        """

        self.__schema = PostgresSchema(prefix=prefix)
        self.connection: Optional[asyncpg.Connection] = None
        self.runtime: Optional[PostgresInteractionRuntime] = None
        self.__session: Optional[AbstractAsyncContextManager[None]] = None
        self.__token: Optional[Token[Optional[PostgresInteractionRuntime]]] = None

    async def __aenter__(self) -> InteractionPostgresSchema:
        """
        Create the schema and initialize the persistent store against it.
        """

        self.connection = await self.__schema.__aenter__()
        self.runtime = PostgresInteractionRuntime(
            configuration=PostgresInteractionConfiguration(
                database="postgres",
                host="localhost",
                password="postgres",
                schema_name=self.__schema.name,
                user="postgres",
                pool_max_size=2,
            )
        )
        await self.runtime.initialize()
        self.__session = self.runtime.session()
        await self.__session.__aenter__()
        self.__token = ACTIVE_RUNTIME.set(self.runtime)
        return self

    async def __aexit__(
        self,
        exception_type: Optional[Type[BaseException]],
        exception: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        """
        Close store connections and drop the disposable schema.
        """

        if self.__token is not None:
            ACTIVE_RUNTIME.reset(self.__token)
            self.__token = None

        if self.__session is not None:
            await self.__session.__aexit__(
                exception_type,
                exception,
                traceback,
            )
            self.__session = None

        if self.runtime is not None:
            await self.runtime.close()
            self.runtime = None
        await self.__schema.__aexit__(
            exception_type=exception_type,
            exception=exception,
            traceback=traceback,
        )
        await LoopDrain.settle()
