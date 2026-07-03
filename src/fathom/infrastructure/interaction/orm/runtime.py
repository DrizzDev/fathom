from __future__ import annotations

from contextlib import asynccontextmanager
from logging import getLogger
from typing import TYPE_CHECKING, Dict, Optional
from urllib.parse import unquote, urlparse

from tortoise import Tortoise
from tortoise.backends.base.client import BaseDBAsyncClient
from tortoise.context import TortoiseContext, _current_context
from tortoise.transactions import in_transaction

from fathom.constants.storage import (
    INTERACTION_POSTGRES_ORM_APP,
    INTERACTION_POSTGRES_ORM_CONNECTION,
    INTERACTION_POSTGRES_ORM_ENGINE,
    InteractionBackend,
)
from fathom.core.exceptions import InteractionError, StorageConfigurationError
from fathom.infrastructure.interaction.orm.models import Catalog
from fathom.infrastructure.interaction.orm.observation import QueryObserver
from fathom.schemas.configuration import PostgresInteractionConfiguration

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator
    from logging import Logger


class PostgresConnectionTarget:
    """
    Parsed Postgres connection target for interaction-store runtime setup.
    """

    def __init__(
        self,
        *,
        host: str,
        port: int,
        user: Optional[str],
        password: Optional[str],
        database: str,
    ) -> None:
        """
        Capture validated Postgres connection fields.
        """

        self.__host = host
        self.__port = port
        self.__user = user
        self.__database = database
        self.__password = password

    @property
    def host(self) -> str:
        """
        Return the target host.
        """

        return self.__host

    @property
    def port(self) -> int:
        """
        Return the target port.
        """

        return self.__port

    @property
    def user(self) -> Optional[str]:
        """
        Return the target user.
        """

        return self.__user

    @property
    def password(self) -> Optional[str]:
        """
        Return the target password.
        """

        return self.__password

    @property
    def database(self) -> str:
        """
        Return the target database.
        """

        return self.__database


class PostgresInteractionRuntime:
    """
    Owns database runtime lifecycle for the Postgres interaction store.
    """

    def __init__(
        self,
        *,
        configuration: PostgresInteractionConfiguration,
        logger: Optional[Logger] = None,
    ) -> None:
        """
        Capture the validated interaction-store configuration.
        """

        self.__initialized = False
        self.__context: Optional[TortoiseContext] = None
        self.__logger = logger or getLogger("fathom.infrastructure.interaction.orm")

        self.__configuration = configuration
        self.__observer = QueryObserver(
            logger=self.__logger,
            threshold=configuration.slow_query_threshold,
        )

    @property
    def initialized(self) -> bool:
        """
        Return whether this runtime owns an initialized connection pool.
        """

        return self.__initialized

    async def initialize(self) -> None:
        """
        Initialize the database runtime connection pool.
        """

        if self.__initialized:
            return

        try:
            self.__context = await Tortoise.init(config=self.__runtime_configuration())
            self.__observe_queries()
        except Exception as exception:
            try:
                if self.__context is not None:
                    await self.__context.close_connections()
                    self.__context = None

            except Exception as cleanup_exception:
                exception.add_note(
                    "Failed to close ORM connections after initialization failure: "
                    f"{type(cleanup_exception).__name__}: {cleanup_exception}"
                )
            raise

        self.__initialized = True

    async def close(self) -> None:
        """
        Close the owned database runtime connection pool.
        """

        if not self.__initialized:
            return

        if self.__context is not None:
            await self.__context.close_connections()

            self.__context = None

        self.__initialized = False

    @asynccontextmanager
    async def session(self) -> AsyncGenerator[None, None]:
        """
        Activate the owned ORM context for one adapter operation.
        """

        if not self.__initialized:
            raise InteractionError("Postgres interaction runtime is not initialized.")

        if self.__context is None:
            raise InteractionError("Postgres interaction runtime context is not initialized.")

        token = _current_context.set(self.__context)

        try:
            yield
        finally:
            _current_context.reset(token)

    @asynccontextmanager
    async def transaction(self) -> AsyncGenerator[BaseDBAsyncClient, None]:
        """
        Open one transaction boundary for grouped interaction writes.
        """

        async with (
            self.session(),
            in_transaction(connection_name=INTERACTION_POSTGRES_ORM_CONNECTION) as connection,
        ):
            yield connection

    def connection_target(self) -> PostgresConnectionTarget:
        """
        Return parsed connection fields for the configured Postgres target.
        """

        if self.__configuration.dsn is None:
            if self.__configuration.host is None:
                raise StorageConfigurationError(
                    backend=InteractionBackend.POSTGRES.value,
                    message="Postgres host is required when DSN is not configured.",
                )

            return PostgresConnectionTarget(
                host=self.__configuration.host,
                port=self.__configuration.port,
                user=self.__configuration.user,
                password=self.__configuration.password,
                database=self.__configuration.database,
            )

        parsed = urlparse(self.__configuration.dsn)

        if parsed.scheme not in ("postgres", "postgresql"):
            raise StorageConfigurationError(
                backend=InteractionBackend.POSTGRES.value,
                message=(
                    "Postgres DSN scheme must be 'postgres' or 'postgresql'; "
                    f"got '{parsed.scheme}'."
                ),
            )
        if parsed.hostname is None:
            raise StorageConfigurationError(
                backend=InteractionBackend.POSTGRES.value,
                message="Postgres DSN requires a hostname.",
            )

        if not parsed.path or parsed.path == "/":
            raise StorageConfigurationError(
                backend=InteractionBackend.POSTGRES.value,
                message="Postgres DSN requires a database path.",
            )

        return PostgresConnectionTarget(
            host=parsed.hostname,
            port=parsed.port or self.__configuration.port,
            database=unquote(parsed.path.lstrip("/")),
            user=unquote(parsed.username) if parsed.username is not None else None,
            password=unquote(parsed.password) if parsed.password is not None else None,
        )

    def server_settings(self) -> Dict[str, str]:
        """
        Return server settings required by Postgres connections.
        """

        return {
            "application_name": self.__configuration.application_name,
            "statement_timeout": str(self.__configuration.statement_timeout),
        }

    def __runtime_configuration(self) -> Dict[str, object]:
        """
        Render the vendor runtime configuration at the adapter boundary.
        """

        target = self.connection_target()
        return {
            "connections": {
                "default": {
                    "engine": INTERACTION_POSTGRES_ORM_ENGINE,
                    "credentials": {
                        "host": target.host,
                        "port": target.port,
                        "user": target.user,
                        "password": target.password,
                        "database": target.database,
                        "ssl": self.__configuration.ssl.value,
                        "server_settings": self.server_settings(),
                        "schema": self.__configuration.schema_name,
                        "minsize": self.__configuration.pool_min_size,
                        "maxsize": self.__configuration.pool_max_size,
                        "application_name": self.__configuration.application_name,
                    },
                }
            },
            "apps": {
                INTERACTION_POSTGRES_ORM_APP: {
                    "models": [Catalog.module()],
                    "default_connection": INTERACTION_POSTGRES_ORM_CONNECTION,
                }
            },
        }

    def __observe_queries(self) -> None:
        """
        Attach query observation to the initialized Tortoise connection.
        """

        if self.__context is None:
            return

        connections = getattr(self.__context, "connections", None)
        if connections is None:
            return

        connection = connections.get(INTERACTION_POSTGRES_ORM_CONNECTION)
        if connection is None:
            return

        self.__observer.observe(client=connection)
