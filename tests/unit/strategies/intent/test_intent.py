from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from typing import Any, Callable, Optional, cast
from unittest.mock import AsyncMock, MagicMock

from fathom.adapters.checkpoint import SqliteCheckpointStore
from fathom.constants.events import FathomEvent
from fathom.constants.state import RunOutcome
from fathom.runtime.checkpoint_serde import CheckpointSerdeFactory
from fathom.schemas.checkpoint import SqliteCheckpointPolicy
from fathom.strategies.intent import (
    CHECKPOINT_ALLOWED_JSON_MODULES,
    CHECKPOINT_ALLOWED_MSGPACK_MODULES,
    IntentStrategy,
)


class CheckpointStoreSerdeTest(unittest.IsolatedAsyncioTestCase):
    """
    Cover SqliteCheckpointStore serde wiring through the new checkpoint port.
    """

    @staticmethod
    def __module_available(name: str) -> bool:
        """
        Return True when an optional checkpointer dependency module is importable.
        """

        try:
            return importlib.util.find_spec(name) is not None
        except ModuleNotFoundError:
            return False

    async def test_adapter_passes_fathom_serde_with_allowed_modules(self) -> None:
        """
        SqliteCheckpointStore.open hands LangGraph the CheckpointSerdeFactory-built serde with the fathom allow-list.
        """

        sqlite_available = self.__module_available(
            "langgraph.checkpoint.sqlite.aio"
        ) and self.__module_available("aiosqlite")
        if not sqlite_available:
            self.skipTest("langgraph-checkpoint-sqlite / aiosqlite not installed")

        serde = CheckpointSerdeFactory.build()

        with tempfile.TemporaryDirectory() as directory:
            store = SqliteCheckpointStore(
                directory=Path(directory),
                policy=SqliteCheckpointPolicy(),
                serde=serde,
            )

            async with store.open(workflow_id="workflow-allowlist-test") as checkpointer:
                self.assertEqual(type(checkpointer).__name__, "AsyncSqliteSaver")
                self.assertEqual(type(checkpointer.serde).__name__, "JsonPlusSerializer")
                allowed_modules = getattr(
                    checkpointer.serde,
                    "_allowed_json_modules",
                    checkpointer.serde._allowed_modules,
                )
                self.assertEqual(allowed_modules, set(CHECKPOINT_ALLOWED_JSON_MODULES))
                if hasattr(checkpointer.serde, "_allowed_msgpack_modules"):
                    self.assertEqual(
                        checkpointer.serde._allowed_msgpack_modules,
                        set(CHECKPOINT_ALLOWED_MSGPACK_MODULES),
                    )


class IntentStrategyTest(unittest.IsolatedAsyncioTestCase):
    """
    Cover IntentStrategy SCRIPT_GENERATED emit behavior in isolation.
    """

    @staticmethod
    def __strategy_with_stubbed_context(*, step_count: int = 3) -> Any:
        """
        Build a bare IntentStrategy with just enough graph-context stubs to
        exercise the SCRIPT_GENERATED emit contract in isolation.
        """

        strategy = object.__new__(IntentStrategy)

        agent_state = MagicMock()
        agent_state.step_count = step_count

        telemetry = MagicMock()
        telemetry.info = AsyncMock()

        graph_context = MagicMock()
        graph_context.telemetry = telemetry
        graph_context.agent_state = agent_state

        strategy.__setattr__("_IntentStrategy__graph_context", graph_context)
        strategy.__setattr__("_IntentStrategy__workflow_id", "wf-test")
        return strategy, telemetry

    async def __invoke_emit(
        self,
        *,
        strategy: Any,
        script_data: Optional[str],
        run_outcome: RunOutcome = RunOutcome.COMPLETED,
    ) -> None:
        """
        Call the private SCRIPT_GENERATED emit on the strategy under test.
        """

        emit = cast(
            "Callable[..., Any]",
            strategy.__getattribute__("_IntentStrategy__emit_script_generated_event"),
        )
        await emit(script_data=script_data, run_outcome=run_outcome)

    async def test_script_generated_emits_with_non_empty_content(self) -> None:
        """
        Non-empty script must emit SCRIPT_GENERATED with the content and is_empty=False.
        """

        strategy, telemetry = self.__strategy_with_stubbed_context(step_count=5)
        await self.__invoke_emit(strategy=strategy, script_data="open swiggy\nsearch biryani")

        telemetry.info.assert_awaited_once()
        call = telemetry.info.call_args
        self.assertEqual(call.args[0], "open swiggy\nsearch biryani")
        self.assertEqual(call.kwargs["type"], FathomEvent.SCRIPT_GENERATED)
        self.assertEqual(call.kwargs["step"], 5)
        self.assertFalse(call.kwargs["is_empty"])

    async def test_script_generated_emits_even_when_content_is_empty(self) -> None:
        """
        Regression: empty script must STILL emit SCRIPT_GENERATED with is_empty=True.
        """

        strategy, telemetry = self.__strategy_with_stubbed_context(step_count=7)
        await self.__invoke_emit(strategy=strategy, script_data="")

        telemetry.info.assert_awaited_once()
        call = telemetry.info.call_args
        self.assertEqual(call.args[0], "")
        self.assertEqual(call.kwargs["type"], FathomEvent.SCRIPT_GENERATED)
        self.assertEqual(call.kwargs["step"], 7)
        self.assertTrue(call.kwargs["is_empty"])

    async def test_script_generated_emits_when_content_is_none(self) -> None:
        """
        None script must still emit a terminal event with is_empty=True.
        """

        strategy, telemetry = self.__strategy_with_stubbed_context(step_count=2)
        await self.__invoke_emit(strategy=strategy, script_data=None)

        telemetry.info.assert_awaited_once()
        call = telemetry.info.call_args
        self.assertEqual(call.args[0], "")
        self.assertTrue(call.kwargs["is_empty"])

    async def test_script_generated_treats_whitespace_only_as_empty(self) -> None:
        """
        Whitespace-only script data is not meaningful content; must mark is_empty=True
        while preserving the raw payload for debugging consumers.
        """

        strategy, telemetry = self.__strategy_with_stubbed_context(step_count=1)
        await self.__invoke_emit(strategy=strategy, script_data="   \n\t  ")

        telemetry.info.assert_awaited_once()
        call = telemetry.info.call_args
        self.assertEqual(call.args[0], "   \n\t  ")
        self.assertTrue(call.kwargs["is_empty"])
