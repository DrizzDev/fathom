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
from fathom.adapters.evidence.history import HistoryEvidenceSource
from fathom.constants.authoring import AuthoringArtifactKind, AuthoringStatus
from fathom.constants.dialect import DialectName
from fathom.constants.events import FathomEvent
from fathom.constants.flow import IssueCode
from fathom.constants.generation import ScriptSource, ScriptStatus
from fathom.constants.state import RunOutcome
from fathom.core.services.decomposer import IntentDecomposer
from fathom.core.services.history import HistoryService
from fathom.interfaces.authoring import AuthoringPort
from fathom.interfaces.llm import LLMPort
from fathom.interfaces.memory import MemoryPort
from fathom.runtime.checkpoint_serde import CheckpointSerdeFactory
from fathom.schemas.authoring import AuthoringArtifact, AuthoringResponse, AuthoringTask
from fathom.schemas.checkpoint import SqliteCheckpointPolicy
from fathom.schemas.configuration import FathomConfiguration
from fathom.schemas.flow import Evidence, Issue, RunObjective
from fathom.schemas.generation import BaselineArtifact, ScriptFileMetadata
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


class IntentStrategyFinalDeliveryTest(unittest.IsolatedAsyncioTestCase):
    """
    Cover the completed-run delivery contract: quality, then baseline, then typed failure; never empty success.
    """

    @staticmethod
    def __strategy(*, baseline: BaselineArtifact, step_count: int = 4) -> Any:
        """
        Build a bare IntentStrategy whose history returns the given baseline outcome.
        """

        strategy = object.__new__(IntentStrategy)

        agent_state = MagicMock()
        agent_state.step_count = step_count

        telemetry = MagicMock()
        telemetry.info = AsyncMock()

        history = MagicMock()
        history.read_baseline_outcome = AsyncMock(return_value=baseline)

        graph_context = MagicMock()
        graph_context.telemetry = telemetry
        graph_context.history = history
        graph_context.agent_state = agent_state

        strategy.__setattr__("_IntentStrategy__workflow_id", "wf-test")
        strategy.__setattr__("_IntentStrategy__graph_context", graph_context)

        return strategy, telemetry, history

    @staticmethod
    async def __deliver(*, strategy: Any, quality: Optional[str]) -> None:
        """
        Invoke the private completed-run delivery entry point.
        """

        deliver = cast(
            "Callable[..., Any]",
            strategy.__getattribute__("_IntentStrategy__deliver_final_script"),
        )
        await deliver(quality=quality, run_outcome=RunOutcome.COMPLETED)

    @staticmethod
    def __generated_baseline() -> BaselineArtifact:
        return BaselineArtifact(
            text="OPEN_APP: com.app\nTap on Search",
            metadata=ScriptFileMetadata(
                source=ScriptSource.BASELINE, status=ScriptStatus.GENERATED
            ),
        )

    @staticmethod
    def __failed_baseline() -> BaselineArtifact:
        return BaselineArtifact(
            metadata=ScriptFileMetadata(
                source=ScriptSource.BASELINE,
                status=ScriptStatus.FAILED,
                issues=(Issue(code=IssueCode.BASELINE_UNAVAILABLE, message="no baseline"),),
            )
        )

    async def test_quality_script_emits_generated_and_skips_baseline(self) -> None:
        """
        A non-empty quality script emits SCRIPT_GENERATED tagged quality, without reading the baseline.
        """

        strategy, telemetry, history = self.__strategy(baseline=self.__failed_baseline())

        await self.__deliver(strategy=strategy, quality="open app\ntap search")

        history.read_baseline_outcome.assert_not_awaited()
        call = telemetry.info.call_args
        self.assertEqual(call.kwargs["type"], FathomEvent.SCRIPT_GENERATED)
        self.assertEqual(call.kwargs["source"], ScriptSource.QUALITY.value)
        self.assertFalse(call.kwargs["is_empty"])

    async def test_empty_quality_falls_back_to_generated_baseline(self) -> None:
        """
        An empty quality result emits the baseline as SCRIPT_GENERATED tagged baseline.
        """

        strategy, telemetry, history = self.__strategy(baseline=self.__generated_baseline())

        await self.__deliver(strategy=strategy, quality="")

        history.read_baseline_outcome.assert_awaited_once()
        call = telemetry.info.call_args
        self.assertEqual(call.args[0], "OPEN_APP: com.app\nTap on Search")
        self.assertEqual(call.kwargs["type"], FathomEvent.SCRIPT_GENERATED)
        self.assertEqual(call.kwargs["source"], ScriptSource.BASELINE.value)
        self.assertFalse(call.kwargs["is_empty"])

    async def test_no_quality_no_baseline_emits_failed_with_diagnostics(self) -> None:
        """
        With neither quality nor a generated baseline, a SCRIPT_GENERATION_FAILED event carries diagnostics.
        """

        strategy, telemetry, _ = self.__strategy(baseline=self.__failed_baseline())

        await self.__deliver(strategy=strategy, quality=None)

        call = telemetry.info.call_args
        self.assertEqual(call.kwargs["type"], FathomEvent.SCRIPT_GENERATION_FAILED)
        self.assertEqual(call.kwargs["issues"][0]["code"], IssueCode.BASELINE_UNAVAILABLE.value)

    async def test_failed_path_never_emits_empty_script_generated(self) -> None:
        """
        The failure path must not emit a successful empty SCRIPT_GENERATED event.
        """

        strategy, telemetry, _ = self.__strategy(baseline=self.__failed_baseline())

        await self.__deliver(strategy=strategy, quality="   ")

        emitted = {call.kwargs["type"] for call in telemetry.info.call_args_list}
        self.assertNotIn(FathomEvent.SCRIPT_GENERATED, emitted)
        self.assertIn(FathomEvent.SCRIPT_GENERATION_FAILED, emitted)

    @staticmethod
    def __events(records: List[Any]) -> List[Any]:
        """
        Extract the structured event identifiers from captured log records.
        """

        return [getattr(record, "event", None) for record in records]

    async def test_quality_selection_logs_decision_and_terminal_event(self) -> None:
        """
        Selecting quality logs the decision and the terminal SCRIPT_GENERATED emit.
        """

        strategy, _, history = self.__strategy(baseline=self.__failed_baseline())

        with self.assertLogs(IntentStrategy.__module__, level="INFO") as captured:
            await self.__deliver(strategy=strategy, quality="open app\ntap search")

        events = self.__events(captured.records)
        self.assertIn("script.finalization.quality_selected", events)
        self.assertIn("script.telemetry.generated_emitted", events)
        history.read_baseline_outcome.assert_not_awaited()

    async def test_baseline_fallback_logs_unavailable_then_selected(self) -> None:
        """
        Empty quality logs quality-unavailable then baseline-selected with the terminal emit.
        """

        strategy, _, _ = self.__strategy(baseline=self.__generated_baseline())

        with self.assertLogs(IntentStrategy.__module__, level="INFO") as captured:
            await self.__deliver(strategy=strategy, quality=None)

        events = self.__events(captured.records)
        self.assertIn("script.finalization.quality_unavailable", events)
        self.assertIn("script.finalization.baseline_selected", events)
        self.assertIn("script.telemetry.generated_emitted", events)

    async def test_failure_path_logs_decision_and_failed_terminal_event(self) -> None:
        """
        No quality and no generated baseline logs the failure decision and the failed terminal emit.
        """

        strategy, _, _ = self.__strategy(baseline=self.__failed_baseline())

        with self.assertLogs(IntentStrategy.__module__, level="INFO") as captured:
            await self.__deliver(strategy=strategy, quality=None)

        events = self.__events(captured.records)
        self.assertIn("script.finalization.failed", events)
        self.assertIn("script.telemetry.failed_emitted", events)

        failed = next(
            record
            for record in captured.records
            if getattr(record, "event", None) == "script.finalization.failed"
        )
        self.assertIn(IssueCode.BASELINE_UNAVAILABLE.value, failed.__dict__["script.issue_codes"])


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
            if event.get("phase") == "fathom.finalization.history.script"
        ]

    @staticmethod
    async def __baseline(
        *,
        history: HistoryService,
        step_number: int,
    ) -> BaselineArtifact:
        """
        Return a generated baseline through the patched history service method.
        """

        _ = (history, step_number)
        return BaselineArtifact(
            text="tap continue",
            metadata=ScriptFileMetadata(
                source=ScriptSource.BASELINE,
                status=ScriptStatus.GENERATED,
            ),
        )

    @staticmethod
    async def __quality(
        *,
        intent: str,
        step_number: int,
    ) -> str:
        """
        Return a generated quality script through the authoring source.
        """

        _ = (intent, step_number)
        return "quality tap continue"

    @staticmethod
    async def __empty_quality(
        *,
        intent: str,
        step_number: int,
    ) -> str:
        """
        Return no quality script so the finalizer must fall back to the baseline.
        """

        _ = (intent, step_number)
        return ""

    @staticmethod
    async def __slow_quality(
        *,
        intent: str,
        step_number: int,
    ) -> str:
        """
        Delay quality generation long enough for finalization heartbeat coverage.
        """

        _ = (intent, step_number)
        await asyncio.sleep(0.6)
        return "quality tap done"

    @staticmethod
    async def __broken_quality(
        *,
        intent: str,
        step_number: int,
    ) -> str:
        """
        Raise a quality generation failure through the patched history service method.
        """

        _ = (intent, step_number)
        raise RuntimeError("quality generation broke")

    @staticmethod
    async def __blocked_quality(
        *,
        intent: str,
        step_number: int,
    ) -> str:
        """
        Block until the caller cancellation path forces host task cancellation.
        """

        _ = (intent, step_number)
        if TestIntentStrategyCancelledScriptDelivery.__blocked_script_started is not None:
            TestIntentStrategyCancelledScriptDelivery.__blocked_script_started.set()

        wait_forever: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        return await wait_forever

    @staticmethod
    async def __failed_baseline(
        *,
        history: HistoryService,
        step_number: int,
    ) -> BaselineArtifact:
        """
        Return a failed baseline through the patched history service method.
        """

        _ = (history, step_number)
        return BaselineArtifact(
            metadata=ScriptFileMetadata(
                status=ScriptStatus.FAILED,
                source=ScriptSource.BASELINE,
                issues=(Issue(code=IssueCode.BASELINE_UNAVAILABLE, message="no baseline"),),
            )
        )

    @staticmethod
    async def __broken_baseline(
        *,
        history: HistoryService,
        step_number: int,
    ) -> BaselineArtifact:
        """
        Raise a baseline read failure through the patched history service method.
        """

        _ = (history, step_number)
        raise RuntimeError("baseline read broke")

    def __patch_boundaries(
        self,
        *,
        graph: TerminalIntentGraph,
        monkeypatch: pytest.MonkeyPatch,
        baseline: Callable[..., Awaitable[BaselineArtifact]],
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
            "read_baseline_outcome",
            self.__history_baseline_method(baseline=baseline),
        )
        monkeypatch.setattr(
            HistoryEvidenceSource,
            "read",
            self.__history_evidence_method(),
        )

    @staticmethod
    def __history_baseline_method(
        *,
        baseline: Callable[..., Awaitable[BaselineArtifact]],
    ) -> Callable[..., Awaitable[BaselineArtifact]]:
        """
        Adapt a keyword-only baseline helper to the HistoryService instance-method contract.
        """

        async def read_baseline_outcome(
            history: HistoryService,
            *,
            step_number: int,
        ) -> BaselineArtifact:
            """
            Call the baseline helper with explicit keyword arguments.
            """

            return await baseline(history=history, step_number=step_number)

        return read_baseline_outcome

    class __QualityAuthoring(AuthoringPort):
        """
        Authoring source test double for final-script quality generation.
        """

        def __init__(self, *, quality: Callable[..., Awaitable[str]]) -> None:
            """
            Store the configured quality helper.
            """

            self.__quality = quality

        async def author(self, *, task: AuthoringTask) -> AuthoringResponse:
            """
            Return the configured authoring response for a task.
            """

            script = await self.__quality(
                intent=task.intent,
                step_number=task.step_number,
            )
            return AuthoringResponse(
                status=AuthoringStatus.GENERATED,
                artifact=AuthoringArtifact(
                    dialect=DialectName.DRIZZ,
                    kind=AuthoringArtifactKind.TEXT,
                    content=script,
                ),
            )

    @staticmethod
    def __history_evidence_method() -> Callable[..., Awaitable[Evidence]]:
        """
        Adapt the evidence source to deterministic test evidence.
        """

        async def read(
            source: HistoryEvidenceSource,
            *,
            run: str,
            objective: RunObjective,
        ) -> Evidence:
            """
            Return minimal normalized evidence for authoring tests.
            """

            _ = source
            return Evidence(
                intent=objective.intent,
                goal=objective.intent,
                package=objective.package,
                artifacts=(run,),
            )

        return read

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
        quality: Optional[Callable[..., Awaitable[str]]] = None,
        baseline: Optional[Callable[..., Awaitable[BaselineArtifact]]] = None,
    ) -> IntentStrategyHarness:
        """
        Execute IntentStrategy with controlled external boundaries.
        """

        self.__patch_boundaries(
            graph=graph,
            monkeypatch=monkeypatch,
            baseline=baseline or self.__baseline,
        )

        harness = IntentStrategyHarnessBuilder.build(
            tmp_path=tmp_path,
            llm=llm_port_stub,
            memory=memory_port_stub,
            configuration=configuration or IntentCancellationConfigurationBuilder.build(),
            authoring=self.__QualityAuthoring(quality=quality or self.__quality),
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
        assert script_event["message"] == "quality tap continue"
        assert script_event["source"] == ScriptSource.QUALITY.value
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
            baseline=self.__baseline,
            graph=TerminalIntentGraph.host_cancelled(),
        )
        harness = IntentStrategyHarnessBuilder.build(
            tmp_path=tmp_path,
            llm=llm_port_stub,
            memory=memory_port_stub,
            configuration=IntentCancellationConfigurationBuilder.build(),
            authoring=self.__QualityAuthoring(quality=self.__quality),
        )

        with pytest.raises(asyncio.CancelledError):
            await harness.strategy.execute()

        script_events = harness.telemetry.of_type(FathomEvent.SCRIPT_GENERATED)

        assert len(script_events) == 1
        assert script_events[0]["message"] == "quality tap continue"
        assert script_events[0]["source"] == ScriptSource.QUALITY.value
        assert script_events[0]["run_outcome"] == RunOutcome.CANCELLED.value

    async def test_host_cancelled_run_baseline_exception_emits_failed_event_after_empty_quality(
        self,
        tmp_path: Path,
        llm_port_stub: LLMPort,
        memory_port_stub: MemoryPort,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """
        Host-level cancellation emits a typed failed event when quality is empty and baseline fails.
        """

        self.__patch_boundaries(
            monkeypatch=monkeypatch,
            baseline=self.__broken_baseline,
            graph=TerminalIntentGraph.host_cancelled(),
        )
        harness = IntentStrategyHarnessBuilder.build(
            tmp_path=tmp_path,
            llm=llm_port_stub,
            memory=memory_port_stub,
            configuration=IntentCancellationConfigurationBuilder.build(),
            authoring=self.__QualityAuthoring(quality=self.__empty_quality),
        )

        with (
            self.__capture_intent_logs() as captured_logs,
            pytest.raises(asyncio.CancelledError),
        ):
            await harness.strategy.execute()

        failed_event = harness.telemetry.of_type(FathomEvent.SCRIPT_GENERATION_FAILED)[0]

        assert failed_event["source"] == ScriptSource.BASELINE.value
        assert failed_event["run_outcome"] == RunOutcome.CANCELLED.value
        assert failed_event["issues"][0]["code"] == IssueCode.BASELINE_UNAVAILABLE.value
        assert any(
            "cancelled-run script fallback failed" in record.message
            for record in captured_logs.records
        )

    async def test_cancelled_run_quality_timeout_falls_back_to_baseline(
        self,
        tmp_path: Path,
        llm_port_stub: LLMPort,
        memory_port_stub: MemoryPort,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """
        A cancelled-run quality timeout falls back to the deterministic baseline.
        """

        harness = await self.__run_strategy(
            tmp_path=tmp_path,
            monkeypatch=monkeypatch,
            llm_port_stub=llm_port_stub,
            baseline=self.__baseline,
            quality=self.__slow_quality,
            memory_port_stub=memory_port_stub,
            graph=TerminalIntentGraph.workflow_cancelled(),
            configuration=IntentCancellationConfigurationBuilder.build(script_timeout=0.1),
        )

        script_event = harness.telemetry.of_type(FathomEvent.SCRIPT_GENERATED)[0]

        assert script_event["message"] == "tap continue"
        assert script_event["source"] == ScriptSource.BASELINE.value
        assert script_event["run_outcome"] == RunOutcome.CANCELLED.value

    async def test_cancelled_run_baseline_exception_emits_failed_event_after_empty_quality(
        self,
        tmp_path: Path,
        llm_port_stub: LLMPort,
        memory_port_stub: MemoryPort,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """
        A cancelled-run baseline exception logs the failure and emits a typed failed event.
        """

        with self.__capture_intent_logs() as captured_logs:
            harness = await self.__run_strategy(
                tmp_path=tmp_path,
                monkeypatch=monkeypatch,
                llm_port_stub=llm_port_stub,
                quality=self.__empty_quality,
                baseline=self.__broken_baseline,
                memory_port_stub=memory_port_stub,
                graph=TerminalIntentGraph.workflow_cancelled(),
            )

        failed_event = harness.telemetry.of_type(FathomEvent.SCRIPT_GENERATION_FAILED)[0]

        assert failed_event["source"] == ScriptSource.BASELINE.value
        assert failed_event["run_outcome"] == RunOutcome.CANCELLED.value
        assert failed_event["issues"][0]["code"] == IssueCode.BASELINE_UNAVAILABLE.value
        assert any(
            "cancelled-run script fallback failed" in record.message
            for record in captured_logs.records
        )

    async def test_slow_cancelled_run_quality_emits_heartbeat_before_script(
        self,
        tmp_path: Path,
        llm_port_stub: LLMPort,
        memory_port_stub: MemoryPort,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """
        Slow cancelled-run quality generation emits a bounded heartbeat before the script event.
        """

        harness = await self.__run_strategy(
            tmp_path=tmp_path,
            monkeypatch=monkeypatch,
            llm_port_stub=llm_port_stub,
            quality=self.__slow_quality,
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
            and event.get("phase") in {None, "fathom.finalization.history.script"}
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
            baseline=self.__baseline,
            graph=TerminalIntentGraph.workflow_cancelled(),
        )
        harness = IntentStrategyHarnessBuilder.build(
            tmp_path=tmp_path,
            llm=llm_port_stub,
            memory=memory_port_stub,
            configuration=IntentCancellationConfigurationBuilder.build(
                script_timeout=1.0, heartbeat_threshold=0.5
            ),
            authoring=self.__QualityAuthoring(quality=self.__slow_quality),
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
        assert script_event["message"] == "quality tap done"
        assert script_event["source"] == ScriptSource.QUALITY.value
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
            baseline=self.__broken_baseline,
            quality=self.__broken_quality,
            memory_port_stub=memory_port_stub,
            graph=TerminalIntentGraph.completed(),
        )

        script_events = harness.telemetry.of_type(FathomEvent.SCRIPT_GENERATED)

        assert script_events == []
        assert harness.strategy.step_results == []

    async def test_completed_run_with_final_authoring_disabled_uses_baseline(
        self,
        tmp_path: Path,
        llm_port_stub: LLMPort,
        memory_port_stub: MemoryPort,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """
        Disabling final authoring skips the quality path and emits the baseline fallback.
        """

        async def fail_if_quality_runs(
            *,
            intent: str,
            step_number: int,
        ) -> str:
            """
            Fail the test if disabled final authoring still calls the quality path.
            """

            _ = (intent, step_number)
            raise AssertionError(
                "quality authoring should not run when final authoring is disabled"
            )

        configuration = IntentCancellationConfigurationBuilder.build()
        configuration = configuration.model_copy(
            update={
                "authoring": configuration.authoring.model_copy(
                    update={
                        "run": configuration.authoring.run.model_copy(update={"enabled": False})
                    }
                )
            }
        )

        harness = await self.__run_strategy(
            tmp_path=tmp_path,
            monkeypatch=monkeypatch,
            baseline=self.__baseline,
            llm_port_stub=llm_port_stub,
            configuration=configuration,
            quality=fail_if_quality_runs,
            memory_port_stub=memory_port_stub,
            graph=TerminalIntentGraph.completed(),
        )

        script_event = harness.telemetry.of_type(FathomEvent.SCRIPT_GENERATED)[0]

        assert script_event["message"] == "tap continue"
        assert script_event["source"] == ScriptSource.BASELINE.value
        assert script_event["run_outcome"] == RunOutcome.COMPLETED.value

    async def test_cancelled_run_blocked_quality_timeout_emits_baseline_not_empty_script(
        self,
        tmp_path: Path,
        llm_port_stub: LLMPort,
        memory_port_stub: MemoryPort,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """
        Blocked quality generation times out and emits the baseline fallback, never empty success.
        """

        self.__patch_boundaries(
            monkeypatch=monkeypatch,
            baseline=self.__baseline,
            graph=TerminalIntentGraph.workflow_cancelled(),
        )
        harness = IntentStrategyHarnessBuilder.build(
            tmp_path=tmp_path,
            llm=llm_port_stub,
            memory=memory_port_stub,
            configuration=IntentCancellationConfigurationBuilder.build(
                script_timeout=1.0, heartbeat_threshold=0.5
            ),
            authoring=self.__QualityAuthoring(quality=self.__blocked_quality),
        )

        TestIntentStrategyCancelledScriptDelivery.__blocked_script_started = asyncio.Event()
        task = asyncio.create_task(harness.strategy.execute())
        await asyncio.wait_for(
            TestIntentStrategyCancelledScriptDelivery.__blocked_script_started.wait(),
            timeout=1.0,
        )
        try:
            await task
        finally:
            TestIntentStrategyCancelledScriptDelivery.__blocked_script_started = None

        script_event = harness.telemetry.of_type(FathomEvent.SCRIPT_GENERATED)[0]

        assert script_event["is_empty"] is False
        assert script_event["message"] == "tap continue"
        assert script_event["source"] == ScriptSource.BASELINE.value
        assert script_event["run_outcome"] == RunOutcome.CANCELLED.value
