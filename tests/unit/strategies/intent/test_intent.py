from __future__ import annotations

import asyncio
import importlib.util
import logging
import tempfile
import unittest
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Dict, List, Optional, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from fathom.adapters.checkpoint import SqliteCheckpointStore
from fathom.constants.events import FathomEvent
from fathom.constants.finalization import FinalizationPhase
from fathom.constants.state import RunOutcome
from fathom.core.services.decomposer import IntentDecomposer
from fathom.core.services.history import HistoryService
from fathom.interfaces.llm import LLMPort
from fathom.interfaces.memory import MemoryPort
from fathom.runtime.checkpoint_serde import CheckpointSerdeFactory
from fathom.schemas.checkpoint import SqliteCheckpointPolicy
from fathom.schemas.configuration import FathomConfiguration
from fathom.strategies.graph.intent.builder import IntentGraphBuilder
from fathom.strategies.intent import (
    CHECKPOINT_ALLOWED_JSON_MODULES,
    CHECKPOINT_ALLOWED_MSGPACK_MODULES,
    IntentStrategy,
)
from tests.builders.intent import (
    DeterministicDecomposer,
    IntentCancellationConfigurationBuilder,
    IntentStrategyHarness,
    IntentStrategyHarnessBuilder,
    TerminalIntentGraph,
)

if TYPE_CHECKING:
    from types import TracebackType


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

        strategy.__setattr__("_IntentStrategy__workflow_id", "wf-test")
        strategy.__setattr__("_IntentStrategy__graph_context", graph_context)

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


@pytest.mark.asyncio
class TestIntentStrategyCancelledScriptDelivery:
    """
    Cover cancelled-run script delivery through the real IntentStrategy execution path.
    """

    __blocked_script_started: Optional[asyncio.Event] = None

    class __CapturedLogs(logging.Handler):
        """
        Captures log records from a named logger for assertions.
        """

        def __init__(self, *, logger_name: str) -> None:
            """
            Initialize the log capture handler.
            """

            super().__init__()
            self.records: List[logging.LogRecord] = []
            self.__logger = logging.getLogger(logger_name)
            self.__previous_level = self.__logger.level

        def __enter__(self) -> TestIntentStrategyCancelledScriptDelivery.__CapturedLogs:
            """
            Attach the handler and enable debug-or-higher capture.
            """

            self.__logger.addHandler(self)
            self.__logger.setLevel(logging.DEBUG)
            return self

        def __exit__(
            self,
            exception_type: Optional[type[BaseException]],
            exception: Optional[BaseException],
            traceback: Optional[TracebackType],
        ) -> None:
            """
            Detach the handler and restore logger level.
            """

            _ = (exception_type, exception, traceback)
            self.__logger.removeHandler(self)
            self.__logger.setLevel(self.__previous_level)

        def emit(self, record: logging.LogRecord) -> None:
            """
            Store one emitted log record.
            """

            self.records.append(record)

    @staticmethod
    def __capture_intent_logs() -> TestIntentStrategyCancelledScriptDelivery.__CapturedLogs:
        """
        Capture logs emitted by IntentStrategy.
        """

        return TestIntentStrategyCancelledScriptDelivery.__CapturedLogs(
            logger_name="fathom.strategies.intent"
        )

    @staticmethod
    def __history_script_heartbeat_events(
        *, harness: IntentStrategyHarness
    ) -> List[Dict[str, object]]:
        """
        Return cancelled-script heartbeat events only.
        """

        return [
            event
            for event in harness.telemetry.of_type(FathomEvent.PHASE_HEARTBEAT)
            if event.get("phase") == FinalizationPhase.HISTORY_SCRIPT.value
        ]

    @staticmethod
    async def __script(
        *,
        history: HistoryService,
        intent: str,
        step_number: int,
    ) -> str:
        """
        Return a partial script through the patched history service method.
        """

        _ = (history, intent, step_number)
        return "tap continue"

    @staticmethod
    async def __empty_script(
        *,
        history: HistoryService,
        intent: str,
        step_number: int,
    ) -> str:
        """
        Return an empty partial script through the patched history service method.
        """

        _ = (history, intent, step_number)
        return ""

    @staticmethod
    async def __broken_script(
        *,
        history: HistoryService,
        intent: str,
        step_number: int,
    ) -> str:
        """
        Raise a script export failure through the patched history service method.
        """

        _ = (history, intent, step_number)
        raise RuntimeError("exporter broke")

    @staticmethod
    async def __slow_script(
        *,
        history: HistoryService,
        intent: str,
        step_number: int,
    ) -> str:
        """
        Delay script generation long enough for finalization heartbeat coverage.
        """

        _ = (history, intent, step_number)
        await asyncio.sleep(0.6)
        return "tap done"

    @staticmethod
    async def __blocked_script(
        *,
        history: HistoryService,
        intent: str,
        step_number: int,
    ) -> str:
        """
        Block until the caller cancellation path forces the script timeout.
        """

        _ = (history, intent, step_number)
        if TestIntentStrategyCancelledScriptDelivery.__blocked_script_started is not None:
            TestIntentStrategyCancelledScriptDelivery.__blocked_script_started.set()

        wait_forever: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        return await wait_forever

    def __patch_boundaries(
        self,
        *,
        graph: TerminalIntentGraph,
        monkeypatch: pytest.MonkeyPatch,
        script: Callable[..., Awaitable[str]],
    ) -> None:
        """
        Patch external graph and script boundaries while keeping real application classes in use.
        """

        monkeypatch.setattr(
            IntentDecomposer,
            "with_configuration",
            staticmethod(lambda **_: DeterministicDecomposer()),
        )
        monkeypatch.setattr(
            IntentGraphBuilder,
            "build",
            lambda builder, *, checkpointer, interrupt_before: self.__selected_graph(
                graph=graph,
                builder=builder,
                checkpointer=checkpointer,
                interrupt_before=interrupt_before,
            ),
        )
        monkeypatch.setattr(
            HistoryService,
            "get_current_script",
            self.__history_script_method(script=script),
        )

    @staticmethod
    def __history_script_method(
        *,
        script: Callable[..., Awaitable[str]],
    ) -> Callable[..., Awaitable[str]]:
        """
        Adapt a keyword-only script helper to the HistoryService instance-method contract.
        """

        async def get_current_script(
            history: HistoryService,
            *,
            intent: str,
            step_number: int,
        ) -> str:
            """
            Call the script helper with explicit keyword arguments.
            """

            return await script(history=history, intent=intent, step_number=step_number)

        return get_current_script

    @staticmethod
    def __selected_graph(
        builder: IntentGraphBuilder,
        *,
        checkpointer: object,
        interrupt_before: List[str],
        graph: TerminalIntentGraph,
    ) -> TerminalIntentGraph:
        """
        Return the selected deterministic graph while accepting the real builder signature.
        """

        _ = (builder, checkpointer, interrupt_before)
        return graph

    async def __run_strategy(
        self,
        *,
        tmp_path: Path,
        llm_port_stub: LLMPort,
        graph: TerminalIntentGraph,
        memory_port_stub: MemoryPort,
        monkeypatch: pytest.MonkeyPatch,
        configuration: Optional[FathomConfiguration] = None,
        script: Optional[Callable[..., Awaitable[str]]] = None,
    ) -> IntentStrategyHarness:
        """
        Execute IntentStrategy with controlled external boundaries.
        """

        self.__patch_boundaries(
            graph=graph,
            monkeypatch=monkeypatch,
            script=script or self.__script,
        )

        harness = IntentStrategyHarnessBuilder.build(
            tmp_path=tmp_path,
            llm=llm_port_stub,
            memory=memory_port_stub,
            configuration=configuration or IntentCancellationConfigurationBuilder.build(),
        )

        await harness.strategy.execute()
        return harness

    async def test_workflow_cancelled_run_emits_partial_script(
        self,
        tmp_path: Path,
        llm_port_stub: LLMPort,
        memory_port_stub: MemoryPort,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """
        Cooperative workflow cancellation must emit the partial script.
        """

        harness = await self.__run_strategy(
            tmp_path=tmp_path,
            monkeypatch=monkeypatch,
            llm_port_stub=llm_port_stub,
            memory_port_stub=memory_port_stub,
            graph=TerminalIntentGraph.workflow_cancelled(),
        )

        script_event = harness.telemetry.of_type(FathomEvent.SCRIPT_GENERATED)[0]

        assert script_event["is_empty"] is False
        assert script_event["message"] == "tap continue"
        assert script_event["run_outcome"] == RunOutcome.CANCELLED.value
        assert script_event["workflow_id"] == "workflow-cancelled-script"

    async def test_host_cancelled_run_emits_partial_script_before_propagating_cancellation(
        self,
        tmp_path: Path,
        llm_port_stub: LLMPort,
        memory_port_stub: MemoryPort,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """
        Host-level task cancellation must still finalize the partial script before propagating.
        """

        self.__patch_boundaries(
            monkeypatch=monkeypatch,
            graph=TerminalIntentGraph.host_cancelled(),
            script=self.__script,
        )
        harness = IntentStrategyHarnessBuilder.build(
            tmp_path=tmp_path,
            llm=llm_port_stub,
            memory=memory_port_stub,
            configuration=IntentCancellationConfigurationBuilder.build(),
        )

        with pytest.raises(asyncio.CancelledError):
            await harness.strategy.execute()

        script_events = harness.telemetry.of_type(FathomEvent.SCRIPT_GENERATED)

        assert len(script_events) == 1
        assert script_events[0]["message"] == "tap continue"
        assert script_events[0]["run_outcome"] == RunOutcome.CANCELLED.value

    async def test_host_cancelled_run_script_exception_emits_empty_script(
        self,
        tmp_path: Path,
        llm_port_stub: LLMPort,
        memory_port_stub: MemoryPort,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """
        Host-level cancellation still emits an empty script event when script export fails.
        """

        self.__patch_boundaries(
            monkeypatch=monkeypatch,
            graph=TerminalIntentGraph.host_cancelled(),
            script=self.__broken_script,
        )
        harness = IntentStrategyHarnessBuilder.build(
            tmp_path=tmp_path,
            llm=llm_port_stub,
            memory=memory_port_stub,
            configuration=IntentCancellationConfigurationBuilder.build(),
        )

        with (
            self.__capture_intent_logs() as captured_logs,
            pytest.raises(asyncio.CancelledError),
        ):
            await harness.strategy.execute()

        script_event = harness.telemetry.of_type(FathomEvent.SCRIPT_GENERATED)[0]

        assert script_event["message"] == ""
        assert script_event["is_empty"] is True
        assert script_event["run_outcome"] == RunOutcome.CANCELLED.value
        assert any(
            "cancelled-run script finalization failed" in record.message
            for record in captured_logs.records
        )

    async def test_cancelled_run_script_timeout_emits_empty_script(
        self,
        tmp_path: Path,
        llm_port_stub: LLMPort,
        memory_port_stub: MemoryPort,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """
        A cancelled-run script timeout emits an empty SCRIPT_GENERATED event tagged cancelled.
        """

        harness = await self.__run_strategy(
            tmp_path=tmp_path,
            monkeypatch=monkeypatch,
            script=self.__slow_script,
            llm_port_stub=llm_port_stub,
            memory_port_stub=memory_port_stub,
            graph=TerminalIntentGraph.workflow_cancelled(),
            configuration=IntentCancellationConfigurationBuilder.build(script_timeout=0.1),
        )

        script_event = harness.telemetry.of_type(FathomEvent.SCRIPT_GENERATED)[0]

        assert script_event["message"] == ""
        assert script_event["is_empty"] is True
        assert script_event["run_outcome"] == RunOutcome.CANCELLED.value

    async def test_cancelled_run_script_exception_emits_empty_script(
        self,
        tmp_path: Path,
        llm_port_stub: LLMPort,
        memory_port_stub: MemoryPort,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """
        A cancelled-run script exception logs the failure and still emits an empty script event.
        """

        with self.__capture_intent_logs() as captured_logs:
            harness = await self.__run_strategy(
                tmp_path=tmp_path,
                monkeypatch=monkeypatch,
                llm_port_stub=llm_port_stub,
                script=self.__broken_script,
                memory_port_stub=memory_port_stub,
                graph=TerminalIntentGraph.workflow_cancelled(),
            )

        script_event = harness.telemetry.of_type(FathomEvent.SCRIPT_GENERATED)[0]

        assert script_event["message"] == ""
        assert script_event["is_empty"] is True
        assert script_event["run_outcome"] == RunOutcome.CANCELLED.value
        assert any(
            "cancelled-run script finalization failed" in record.message
            for record in captured_logs.records
        )

    async def test_slow_cancelled_run_script_emits_heartbeat_before_script(
        self,
        tmp_path: Path,
        llm_port_stub: LLMPort,
        memory_port_stub: MemoryPort,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """
        Slow cancelled-run script generation emits a bounded heartbeat before the script event.
        """

        harness = await self.__run_strategy(
            tmp_path=tmp_path,
            monkeypatch=monkeypatch,
            script=self.__slow_script,
            llm_port_stub=llm_port_stub,
            memory_port_stub=memory_port_stub,
            graph=TerminalIntentGraph.workflow_cancelled(),
            configuration=IntentCancellationConfigurationBuilder.build(
                script_timeout=1.0, heartbeat_threshold=0.5
            ),
        )

        heartbeat_event = self.__history_script_heartbeat_events(harness=harness)[0]
        event_types = [
            event.get("type")
            for event in harness.telemetry.events
            if event.get("type")
            in {
                FathomEvent.PHASE_HEARTBEAT,
                FathomEvent.SCRIPT_GENERATED,
            }
            and event.get("phase") in {None, FinalizationPhase.HISTORY_SCRIPT.value}
        ]

        assert event_types.index(FathomEvent.PHASE_HEARTBEAT) < event_types.index(
            FathomEvent.SCRIPT_GENERATED
        )

        assert heartbeat_event["step"] == 0
        assert heartbeat_event["message"] == "Finalizing the script..."
        assert heartbeat_event["run_outcome"] == RunOutcome.CANCELLED.value
        assert heartbeat_event["workflow_id"] == "workflow-cancelled-script"

    async def test_cancelled_run_heartbeat_failure_does_not_flip_outcome(
        self,
        tmp_path: Path,
        llm_port_stub: LLMPort,
        memory_port_stub: MemoryPort,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """
        Heartbeat telemetry failures must not convert a cancelled run into a failed run.
        """

        self.__patch_boundaries(
            monkeypatch=monkeypatch,
            graph=TerminalIntentGraph.workflow_cancelled(),
            script=self.__slow_script,
        )
        harness = IntentStrategyHarnessBuilder.build(
            tmp_path=tmp_path,
            llm=llm_port_stub,
            memory=memory_port_stub,
            configuration=IntentCancellationConfigurationBuilder.build(
                script_timeout=1.0, heartbeat_threshold=0.5
            ),
        )
        original_info = harness.telemetry.info

        async def fail_only_heartbeat(message: str, **context: object) -> None:
            """
            Raise only for the finalization heartbeat event.
            """

            if context.get("type") == FathomEvent.PHASE_HEARTBEAT:
                raise RuntimeError("telemetry heartbeat broke")

            await original_info(message, **context)

        monkeypatch.setattr(harness.telemetry, "info", fail_only_heartbeat)

        with self.__capture_intent_logs() as captured_logs:
            result = await harness.strategy.execute()
        script_event = harness.telemetry.of_type(FathomEvent.SCRIPT_GENERATED)[0]

        assert result.is_cancelled is True
        assert script_event["message"] == "tap done"
        assert script_event["run_outcome"] == RunOutcome.CANCELLED.value
        assert any(
            "cancelled-run script heartbeat emit failed" in record.message
            for record in captured_logs.records
        )

    async def test_completed_run_script_exception_fails_without_empty_script(
        self,
        tmp_path: Path,
        llm_port_stub: LLMPort,
        memory_port_stub: MemoryPort,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """
        Completed-run script exceptions remain strict and do not emit empty success artifacts.
        """

        harness = await self.__run_strategy(
            tmp_path=tmp_path,
            monkeypatch=monkeypatch,
            llm_port_stub=llm_port_stub,
            script=self.__broken_script,
            memory_port_stub=memory_port_stub,
            graph=TerminalIntentGraph.completed(),
        )

        script_events = harness.telemetry.of_type(FathomEvent.SCRIPT_GENERATED)

        assert harness.strategy.step_results == []
        assert script_events == []

    async def test_cancelled_run_task_cancellation_during_script_generation_emits_empty_script(
        self,
        tmp_path: Path,
        llm_port_stub: LLMPort,
        memory_port_stub: MemoryPort,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """
        Host task cancellation during script generation must propagate after empty script delivery.
        """

        self.__patch_boundaries(
            monkeypatch=monkeypatch,
            graph=TerminalIntentGraph.workflow_cancelled(),
            script=self.__blocked_script,
        )
        harness = IntentStrategyHarnessBuilder.build(
            tmp_path=tmp_path,
            llm=llm_port_stub,
            memory=memory_port_stub,
            configuration=IntentCancellationConfigurationBuilder.build(
                script_timeout=1.0, heartbeat_threshold=0.5
            ),
        )

        TestIntentStrategyCancelledScriptDelivery.__blocked_script_started = asyncio.Event()
        task = asyncio.create_task(harness.strategy.execute())
        await asyncio.wait_for(
            TestIntentStrategyCancelledScriptDelivery.__blocked_script_started.wait(),
            timeout=1.0,
        )
        task.cancel()

        try:
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            TestIntentStrategyCancelledScriptDelivery.__blocked_script_started = None

        script_event = harness.telemetry.of_type(FathomEvent.SCRIPT_GENERATED)[0]

        assert script_event["message"] == ""
        assert script_event["is_empty"] is True
        assert script_event["run_outcome"] == RunOutcome.CANCELLED.value
