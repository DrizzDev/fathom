from __future__ import annotations

import asyncio
import hashlib
import json
import re
from contextlib import asynccontextmanager
from contextvars import ContextVar, Token
from logging import getLogger
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    Dict,
    Final,
    List,
    Optional,
    Protocol,
    Tuple,
    Type,
    Union,
)

import asyncpg

from fathom.core.exceptions import InteractionError
from fathom.infrastructure.interaction.pypika.postgres import schema
from fathom.infrastructure.interaction.timing import SlowQueryLogger, TimedConnection
from fathom.schemas.configuration import PostgresInteractionConfiguration
from fathom.schemas.postgres import PostgresMigrationStep

if TYPE_CHECKING:
    from collections.abc import Generator

    class AsyncpgRecord(Protocol):
        """
        Minimal asyncpg record shape returned by migration queries.
        """

        def __getitem__(self, key: str | int) -> object:
            """
            Return one column value by name.
            """

            ...

    class AsyncpgTransaction(Protocol):
        """
        Async context manager returned by asyncpg transactions.
        """

        async def __aenter__(self) -> object:
            """
            Enter the transaction.
            """

            ...

        async def __aexit__(self, *args: object) -> None:
            """
            Leave the transaction.
            """

            ...

    class AsyncpgConnection(Protocol):
        """
        Minimal asyncpg connection surface used by the unit of work.
        """

        async def fetch(self, sql: str, *parameters: object) -> List[AsyncpgRecord]:
            """
            Execute a query and return all rows.
            """

            ...

        async def fetchrow(self, sql: str, *parameters: object) -> Optional[AsyncpgRecord]:
            """
            Execute a query and return the first row.
            """

            ...

        async def execute(self, sql: str, *parameters: object) -> str:
            """
            Execute a statement and return the command tag.
            """

            ...

        async def set_type_codec(self, *args: object, **kwargs: object) -> None:
            """
            Register a codec with asyncpg.
            """

            ...

        def transaction(self) -> AsyncpgTransaction:
            """
            Create a transaction context manager.
            """

            ...

    class AsyncpgAcquire(Protocol):
        """
        Async context manager returned by pool.acquire().
        """

        async def __aenter__(self) -> AsyncpgConnection:
            """
            Enter the acquired connection context.
            """

            ...

        async def __aexit__(self, *args: object) -> None:
            """
            Release the acquired connection.
            """

            ...

    class AsyncpgPool(Protocol):
        """
        Minimal asyncpg pool surface used by the unit of work.
        """

        def acquire(self) -> AsyncpgAcquire:
            """
            Acquire one connection context manager.
            """

            ...

        async def close(self) -> None:
            """
            Close the pool.
            """

            ...
else:
    AsyncpgPool = object
    AsyncpgRecord = object
    AsyncpgConnection = object  # runtime fallback when asyncpg is uninstalled

logger = getLogger(__name__)


class PostgresCursor:
    """
    Async cursor facade returning materialized asyncpg.Record rows.
    """

    def __init__(self, *, rows: List[AsyncpgRecord]) -> None:
        """
        Store materialized rows and initialise the read position.
        """

        self.__index = 0
        self.__rows = rows

    async def fetchone(self) -> Optional[AsyncpgRecord]:
        """
        Return the next row from the materialized result set.
        """

        if self.__index >= len(self.__rows):
            return None

        row = self.__rows[self.__index]
        self.__index += 1

        return row

    async def fetchall(self) -> List[AsyncpgRecord]:
        """
        Return all remaining rows.
        """

        remaining = self.__rows[self.__index :]
        self.__index = len(self.__rows)

        return remaining


class PostgresExecution:
    """
    Awaitable / async-context execution wrapper used by PostgresConnection.

    Treats the SQL string as opaque Postgres SQL with `$n` placeholders;
    parameters are forwarded to asyncpg as positional arguments.
    """

    def __init__(
        self,
        *,
        sql: str,
        parameters: Tuple[object, ...],
        connection: AsyncpgConnection,
    ) -> None:
        """
        Bind the raw asyncpg connection, SQL text, and bound parameters.
        """

        self.__connection = connection

        self.__sql = sql
        self.rowcount = 0
        self.__parameters = parameters

    def __await__(self) -> "Generator[object, None, PostgresExecution]":
        """
        Execute the statement when awaited directly.
        """

        return self.__execute().__await__()

    async def __aenter__(self) -> PostgresCursor:
        """
        Execute the statement as a query and return a cursor facade.
        """

        rows = await self.__connection.fetch(self.__sql, *self.__parameters)
        self.rowcount = len(rows)

        return PostgresCursor(rows=list(rows))

    async def __aexit__(self, *args: object) -> None:
        """
        Leave the async context without suppressing exceptions.
        """

        return None

    async def __execute(self) -> "PostgresExecution":
        """
        Execute the statement and update rowcount from the command tag.
        """

        status = await self.__connection.execute(self.__sql, *self.__parameters)
        self.rowcount = self.__rowcount(status=status)

        return self

    @staticmethod
    def __rowcount(*, status: str) -> int:
        """
        Extract rowcount from asyncpg command tags such as UPDATE 3.
        """

        token = status.rsplit(" ", 1)[-1]

        if token.isdigit():
            return int(token)

        return 0


class PostgresConnection:
    """
    Thin async wrapper around an asyncpg connection.

    The native Postgres repositories emit `$n` placeholders directly so this
    wrapper passes the SQL through unchanged. The wrapper exists to expose
    the awaitable / async-context shape that the repository code expects.
    """

    def __init__(self, *, connection: AsyncpgConnection) -> None:
        """
        Bind one raw asyncpg connection.
        """

        self.__connection = connection

    def execute(
        self,
        sql: str,
        parameters: Union[Tuple[object, ...], List[object]] = (),
    ) -> PostgresExecution:
        """
        Return an awaitable / context-manager execution object.
        """

        return PostgresExecution(
            sql=sql,
            parameters=tuple(parameters),
            connection=self.__connection,
        )


class Unit:
    """
    Transaction and schema owner for Postgres interaction storage.
    """

    # Namespace half of the (int4, int4) advisory-lock key; isolates Fathom
    # interaction migration locks from other hash text-keyed locks.

    __MIGRATION_LOCK_NAMESPACE: int = 0x46544D31
    __IDENTIFIER_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

    def __init__(self, *, configuration: PostgresInteractionConfiguration) -> None:
        """
        Bind configuration and lazy runtime resources for Postgres storage.
        """

        self.__logger = getLogger(".".join((__name__, self.__class__.__name__)))

        self.__configuration = configuration
        self.__schema = self.__validated_schema(value=configuration.schema_name)

        self.__initialized = False
        self.__lock = asyncio.Lock()
        self.__pool: Optional[AsyncpgPool] = None
        self.__connection: ContextVar[Optional[Any]] = ContextVar(
            "postgres_interaction_connection",
            default=None,
        )
        self.__slow_query_logger = SlowQueryLogger(
            logger=getLogger("fathom.interaction.postgres"),
            threshold_milliseconds=configuration.slow_query_threshold,
            backend="postgres",
        )

    async def initialize(self) -> None:
        """
        Create the asyncpg pool and converge the interaction schema.
        """

        if self.__initialized:
            return

        async with self.__lock:
            if self.__initialized:
                return

            self.__logger.info(
                "Initializing Postgres interaction storage",
                extra=self.__log_extra(
                    event="postgres.interaction.initialize",
                    pool_min_size=self.__configuration.pool_min_size,
                    pool_max_size=self.__configuration.pool_max_size,
                ),
            )
            try:
                pool_kwargs: Dict[str, Any] = {
                    "init": self.__init_connection,
                    "ssl": self.__configuration.ssl.value,
                    "min_size": self.__configuration.pool_min_size,
                    "max_size": self.__configuration.pool_max_size,
                    "server_settings": {
                        "application_name": self.__configuration.application_name,
                        "statement_timeout": str(self.__configuration.statement_timeout),
                    },
                }
                if self.__configuration.dsn is not None:
                    pool_kwargs["dsn"] = self.__configuration.dsn
                else:
                    pool_kwargs["host"] = self.__configuration.host
                    pool_kwargs["port"] = self.__configuration.port
                    pool_kwargs["user"] = self.__configuration.user
                    pool_kwargs["password"] = self.__configuration.password
                    pool_kwargs["database"] = self.__configuration.database

                self.__pool = await asyncpg.create_pool(**pool_kwargs)
                async with self.__pool.acquire() as connection:
                    await connection.execute(f"CREATE SCHEMA IF NOT EXISTS {self.__quoted_schema}")
                    await connection.execute(f"SET search_path TO {self.__quoted_schema}")

                    async with connection.transaction():
                        await self.__acquire_migration_lock(connection=connection)
                        await connection.execute(schema.MIGRATION_TABLE)

                        for step in schema.MIGRATION_STEPS:
                            await self.__apply_migration(connection=connection, step=step)

                    # Run grants outside the ledger so a privilege-restricted
                    # deploy can retry on the next privileged deploy.
                    await self.__apply_bootstrap_grants(connection=connection)
            except asyncio.CancelledError:
                await self.__close_failed_pool()
                raise
            except Exception:
                await self.__close_failed_pool()
                self.__logger.exception(
                    "Postgres interaction storage initialization failed",
                    extra=self.__log_extra(event="postgres.interaction.initialize.failed"),
                )
                raise

            self.__initialized = True
            self.__logger.info(
                "Postgres interaction storage is ready",
                extra=self.__log_extra(
                    event="postgres.interaction.ready",
                    migration_version=schema.SCHEMA_VERSION,
                ),
            )

    async def __close_failed_pool(self) -> None:
        """
        Close and clear a pool created by a failed initialization attempt.
        """

        if self.__pool is None:
            return

        try:
            await self.__pool.close()
        except Exception as close_error:  # noqa: BLE001
            self.__logger.warning(
                "Failed to close Postgres pool after initialization error",
                extra=self.__log_extra(
                    event="postgres.interaction.pool.close.failed",
                    error=type(close_error).__name__,
                ),
            )
        finally:
            self.__pool = None

    @asynccontextmanager
    async def session(self) -> AsyncGenerator[PostgresConnection, None]:
        """
        Yield one Postgres connection inside a managed transaction.
        """

        existing = self.__connection.get()

        if existing is not None:
            yield existing
            return

        await self.initialize()
        pool = self.__require_pool()

        async with pool.acquire() as raw_connection:
            await raw_connection.execute(f"SET search_path TO {self.__quoted_schema}")

            base = PostgresConnection(connection=raw_connection)
            connection: Any = (
                TimedConnection(inner=base, logger=self.__slow_query_logger)
                if self.__slow_query_logger.threshold > 0
                else base
            )
            token: Token[Optional[Any]] = self.__connection.set(connection)

            try:
                async with raw_connection.transaction():
                    yield connection
            except self.__schema_errors() as exception:
                self.__logger.exception(
                    "Postgres interaction schema is incompatible with repository code",
                    extra=self.__log_extra(
                        error_type=type(exception).__name__,
                        event="postgres.interaction.schema.error",
                    ),
                )
                raise InteractionError(
                    "Interaction storage schema is not compatible with this application version. "
                    "Run pending migrations and deploy the matching Fathom package."
                ) from exception

            except self.__integrity_errors() as exception:
                self.__logger.exception(
                    "Postgres interaction transaction failed integrity validation",
                    extra=self.__log_extra(event="postgres.interaction.integrity.error"),
                )
                raise InteractionError(
                    f"Interaction storage integrity violation: {exception}"
                ) from exception

            finally:
                self.__connection.reset(token)

    @asynccontextmanager
    async def atomic(self) -> AsyncGenerator[None, None]:
        """
        Open one grouped transaction boundary that reuses the active session.
        """

        async with self.session():
            yield

    async def close(self) -> None:
        """
        Close the asyncpg pool. Primarily useful for integration tests.
        """

        if self.__pool is not None:
            self.__logger.info(
                "Closing Postgres interaction pool",
                extra=self.__log_extra(event="postgres.interaction.close"),
            )
            await self.__pool.close()

            self.__pool = None
            self.__initialized = False

    @property
    def __quoted_schema(self) -> str:
        """
        Return the validated schema as a quoted SQL identifier.
        """

        return f'"{self.__schema}"'

    @staticmethod
    def __validated_schema(*, value: str) -> str:
        """
        Validate schema_name before using it as an identifier.
        """

        if Unit.__IDENTIFIER_PATTERN.fullmatch(value) is None:
            raise InteractionError(
                "Postgres interaction schema_name must be a simple SQL identifier."
            )

        return value

    def __require_pool(self) -> AsyncpgPool:
        """
        Return the initialized asyncpg pool or fail with diagnostic context.
        """

        if self.__pool is None:
            raise InteractionError("Postgres interaction pool was not initialized.")

        return self.__pool

    @staticmethod
    async def __init_connection(connection: AsyncpgConnection) -> None:
        """
        Register a JSON codec on a freshly-acquired pool connection.

        With the codec installed, the native repositories can pass Python
        dict / list values directly into JSONB columns and read native
        dict / list values back without per-call json.dumps / loads calls.
        """

        await connection.set_type_codec(
            "jsonb",
            encoder=json.dumps,
            decoder=json.loads,
            schema="pg_catalog",
        )
        await connection.set_type_codec(
            "json",
            encoder=json.dumps,
            decoder=json.loads,
            schema="pg_catalog",
        )

    async def __apply_migration(
        self,
        *,
        step: PostgresMigrationStep,
        connection: AsyncpgConnection,
    ) -> None:
        """
        Apply one append-only migration after validating its checksum.
        """

        checksum = self.__migration_checksum(step=step)

        row = await connection.fetchrow(
            "SELECT checksum FROM migrations WHERE version = $1",
            step.version,
        )

        if row is not None and str(row["checksum"]) != checksum:
            self.__logger.error(
                "Postgres interaction migration checksum mismatch",
                extra=self.__log_extra(
                    event="postgres.interaction.migration.checksum.mismatch",
                    migration_name=step.name,
                    migration_version=step.version,
                ),
            )
            raise InteractionError(
                "Postgres interaction migration checksum mismatch for "
                f"version {step.version}; never edit an applied migration."
            )
        if row is not None:
            self.__logger.debug(
                "Postgres interaction migration already applied",
                extra=self.__log_extra(
                    event="postgres.interaction.migration.skip",
                    migration_name=step.name,
                    migration_version=step.version,
                ),
            )
            return

        self.__logger.info(
            "Applying Postgres interaction migration",
            extra=self.__log_extra(
                event="postgres.interaction.migration.apply",
                migration_name=step.name,
                migration_version=step.version,
            ),
        )
        for statement in step.statements:
            await connection.execute(statement)

        # Tolerate a duplicate (version) only when the recorded checksum
        # matches; mismatch is a real divergence and must fail loudly.
        recorded = await connection.fetchrow(
            """
            INSERT INTO migrations (version, name, checksum)
            VALUES ($1, $2, $3)
            ON CONFLICT (version) DO UPDATE SET version = migrations.version
            RETURNING checksum
            """,
            step.version,
            step.name,
            checksum,
        )
        if recorded is None or str(recorded["checksum"]) != checksum:
            self.__logger.error(
                "Postgres interaction migration ledger checksum diverged",
                extra=self.__log_extra(
                    event="postgres.interaction.migration.ledger.diverged",
                    migration_name=step.name,
                    expected_checksum=checksum,
                    migration_version=step.version,
                    recorded_checksum=str(recorded["checksum"]) if recorded else None,
                ),
            )
            raise InteractionError(
                "Postgres interaction migration ledger diverged after apply: "
                f"version {step.version} recorded with a different checksum. "
                "Another deploy raced the migration; resolve manually before retrying."
            )

    async def __apply_bootstrap_grants(self, *, connection: AsyncpgConnection) -> None:
        """
        Run idempotent role/grant/RLS statements outside the migration ledger.
        """

        for statement in schema.BOOTSTRAP_STEPS:
            await connection.execute(statement)

        self.__logger.info(
            "Postgres interaction bootstrap grants applied",
            extra=self.__log_extra(event="postgres.interaction.bootstrap.grants"),
        )

    async def __acquire_migration_lock(self, *, connection: AsyncpgConnection) -> None:
        """
        Acquire a 64-bit transaction-scoped advisory lock for the schema.
        """

        await connection.execute(
            "SELECT pg_advisory_xact_lock($1::int4, hashtext($2)::int4)",
            self.__MIGRATION_LOCK_NAMESPACE,
            self.__schema,
        )

    def __log_extra(self, *, event: str, **values: object) -> Dict[str, object]:
        """
        Build structured log context for Postgres storage lifecycle events.
        """

        return {
            **values,
            "event": event,
            "schema": self.__schema,
            "component": "fathom_interaction_postgres",
        }

    @staticmethod
    def __migration_checksum(*, step: PostgresMigrationStep) -> str:
        """
        Return a deterministic checksum for one migration step.
        """

        payload = "\n".join((str(step.version), step.name, *step.statements))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def __integrity_errors(self) -> Tuple[Type[Exception], ...]:
        """
        Return asyncpg integrity exception classes caught at transaction boundaries.
        """

        return (
            asyncpg.CheckViolationError,
            asyncpg.UniqueViolationError,
            asyncpg.NotNullViolationError,
            asyncpg.ForeignKeyViolationError,
        )

    def __schema_errors(self) -> Tuple[Type[Exception], ...]:
        """
        Return asyncpg schema mismatch exception classes caught at transaction boundaries.
        """

        return (
            asyncpg.UndefinedTableError,
            asyncpg.UndefinedColumnError,
            asyncpg.InvalidSchemaNameError,
        )
