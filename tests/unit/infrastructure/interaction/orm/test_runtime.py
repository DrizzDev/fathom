from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from fathom.core.exceptions import InteractionError
from fathom.infrastructure.interaction.orm.runtime import PostgresInteractionRuntime
from fathom.schemas.configuration import PostgresInteractionConfiguration


class RuntimeContext:
    """
    Test double for the ORM context owned by the runtime.
    """

    def __init__(self) -> None:
        """
        Initialize observable context lifecycle dependencies.
        """

        self.close_connections = AsyncMock()


class TestPostgresInteractionRuntime:
    """
    Verify ORM runtime connection parsing and failure handling.
    """

    def test_connection_target_decodes_percent_encoded_dsn_fields(self) -> None:
        """
        Encoded DSN credentials and database names must reach asyncpg decoded.
        """

        runtime = PostgresInteractionRuntime(
            configuration=PostgresInteractionConfiguration(
                dsn="postgresql://user%40example:p%40ss@localhost:5544/fathom%2Ddb",
            )
        )

        target = runtime.connection_target()

        assert target.host == "localhost"
        assert target.port == 5544
        assert target.user == "user@example"
        assert target.password == "p@ss"
        assert target.database == "fathom-db"

    async def test_session_activates_context_from_another_task(self) -> None:
        """
        Request tasks must resolve the initialized runtime ORM context.
        """

        runtime = PostgresInteractionRuntime(
            configuration=PostgresInteractionConfiguration(
                dsn="postgresql://localhost/postgres",
            )
        )

        context = RuntimeContext()
        initialize = AsyncMock(return_value=context)
        with patch(
            "fathom.infrastructure.interaction.orm.runtime.Tortoise.init",
            new=initialize,
        ):
            await runtime.initialize()
            call = initialize.await_args
            assert call is not None
            assert "_enable_global_fallback" not in call.kwargs

        async def execute() -> bool:
            """
            Confirm an unrelated task can bind the initialized context.
            """

            async with runtime.session():
                return True

        assert await asyncio.create_task(execute()) is True

    async def test_session_after_close_raises_interaction_error(self) -> None:
        """
        Closed runtimes fail with the interaction error boundary.
        """

        runtime = PostgresInteractionRuntime(
            configuration=PostgresInteractionConfiguration(
                dsn="postgresql://localhost/postgres",
            )
        )

        with pytest.raises(InteractionError, match="not initialized"):
            async with runtime.session():
                pass

    async def test_close_releases_owned_context(self) -> None:
        """
        Closing the runtime must release the initialized ORM context.
        """

        runtime = PostgresInteractionRuntime(
            configuration=PostgresInteractionConfiguration(
                dsn="postgresql://localhost/postgres",
            )
        )
        context = RuntimeContext()

        with patch(
            "fathom.infrastructure.interaction.orm.runtime.Tortoise.init",
            new=AsyncMock(return_value=context),
        ):
            await runtime.initialize()
            await runtime.close()

        context.close_connections.assert_awaited_once_with()
        assert runtime.initialized is False
