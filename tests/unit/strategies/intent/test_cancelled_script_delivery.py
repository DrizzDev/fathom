from __future__ import annotations

import asyncio
from typing import Any, Optional

import pytest
from tests.builders.intent import (
    CancelledExecutor,
    CompletedExecutor,
    DecomposerBoundary,
    IntentHarness,
    TerminalGraph,
    build_intent_strategy,
    cancellation_configuration,
)

from fathom.constants.events import FathomEvent
from fathom.constants.finalization import FinalizationPhase
from fathom.constants.state import RunOutcome
from fathom.core.services.decomposer import IntentDecomposer
from fathom.core.services.history import HistoryService
from fathom.interfaces.llm import LLMPort
from fathom.interfaces.memory import MemoryPort
from fathom.schemas.configuration import FathomConfiguration
from fathom.strategies.graph.intent.builder import IntentGraphBuilder


def _history_script_heartbeat_events(harness: IntentHarness) -> list[dict[str, Any]]:
    """
    Return cancelled-script heartbeat events only.
    """

    return [
        event
        for event in harness.telemetry.of_type(FathomEvent.PHASE_HEARTBEAT)
        if event.get("phase") == FinalizationPhase.HISTORY_SCRIPT.value
    ]


@pytest.fixture(autouse=True)
def _patch_graph_boundaries(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Patch public graph/decomposer boundaries so tests exercise IntentStrategy.execute.
    """

    def build_graph(
        self: IntentGraphBuilder,
        *,
        checkpointer: object,
        interrupt_before: list[str],
    ) -> TerminalGraph:
        """
        Return the terminal graph boundary while accepting the real builder signature.
        """

        _ = (self, checkpointer, interrupt_before)
        return TerminalGraph()

    monkeypatch.setattr(
        IntentDecomposer,
        "with_configuration",
        staticmethod(lambda **_: DecomposerBoundary()),
    )
    monkeypatch.setattr(IntentGraphBuilder, "build", build_graph)


async def _run_harness(
    *,
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    llm_port_stub: LLMPort,
    memory_port_stub: MemoryPort,
    executor: type[object],
    script_result: Optional[str] = "tap continue",
    script_delay: float = 0.0,
    script_error: Optional[BaseException] = None,
    configuration: Optional[FathomConfiguration] = None,
) -> IntentHarness:
    """
    Run IntentStrategy.execute with patched executor and script-generation behavior.
    """

    import fathom.runtime.executor as executor_module

    harness = build_intent_strategy(
        tmp_path=tmp_path,
        llm=llm_port_stub,
        memory=memory_port_stub,
        configuration=configuration or cancellation_configuration(),
    )

    async def get_current_script(
        self: HistoryService,
        *,
        intent: str,
        step_number: int,
    ) -> str:
        """
        Script-generation boundary used by the execute path.
        """

        _ = (self, intent, step_number)
        if script_delay > 0:
            await asyncio.sleep(script_delay)
        if script_error is not None:
            raise script_error
        return script_result or ""

    monkeypatch.setattr(executor_module, "GraphExecutor", executor)
    monkeypatch.setattr(HistoryService, "get_current_script", get_current_script)

    await harness.strategy.execute()
    return harness


@pytest.mark.asyncio
async def test_cancelled_run_emits_script_after_cancellation_ack(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    llm_port_stub: LLMPort,
    memory_port_stub: MemoryPort,
) -> None:
    """
    Cooperative cancellation must acknowledge cancellation before emitting the partial script.
    """

    harness = await _run_harness(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        llm_port_stub=llm_port_stub,
        memory_port_stub=memory_port_stub,
        executor=CancelledExecutor,
        script_result="tap continue",
    )

    event_types = [event.get("type") for event in harness.telemetry.events]
    script_event = harness.telemetry.of_type(FathomEvent.SCRIPT_GENERATED)[0]

    assert event_types.index(FathomEvent.WORKFLOW_CANCELLED) < event_types.index(
        FathomEvent.SCRIPT_GENERATED
    )
    assert script_event["message"] == "tap continue"
    assert script_event["is_empty"] is False
    assert script_event["run_outcome"] == RunOutcome.CANCELLED.value
    assert script_event["workflow_id"] == "workflow-cancelled-script"


@pytest.mark.asyncio
async def test_cancelled_run_script_timeout_emits_empty_script(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    llm_port_stub: LLMPort,
    memory_port_stub: MemoryPort,
) -> None:
    """
    A cancelled-run script timeout emits an empty SCRIPT_GENERATED event tagged cancelled.
    """

    harness = await _run_harness(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        llm_port_stub=llm_port_stub,
        memory_port_stub=memory_port_stub,
        executor=CancelledExecutor,
        script_delay=1.0,
        configuration=cancellation_configuration(script_timeout=0.1),
    )

    script_event = harness.telemetry.of_type(FathomEvent.SCRIPT_GENERATED)[0]

    assert script_event["message"] == ""
    assert script_event["is_empty"] is True
    assert script_event["run_outcome"] == RunOutcome.CANCELLED.value


@pytest.mark.asyncio
async def test_cancelled_run_script_exception_emits_empty_script(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    llm_port_stub: LLMPort,
    memory_port_stub: MemoryPort,
) -> None:
    """
    A cancelled-run script exception logs the failure and still emits an empty script event.
    """

    harness = await _run_harness(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        llm_port_stub=llm_port_stub,
        memory_port_stub=memory_port_stub,
        executor=CancelledExecutor,
        script_error=RuntimeError("exporter broke"),
    )

    script_event = harness.telemetry.of_type(FathomEvent.SCRIPT_GENERATED)[0]

    assert script_event["message"] == ""
    assert script_event["is_empty"] is True
    assert script_event["run_outcome"] == RunOutcome.CANCELLED.value
    assert any(
        "cancelled-run script finalization failed" in record.message for record in caplog.records
    )


@pytest.mark.asyncio
async def test_slow_cancelled_run_script_emits_heartbeat_before_script(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    llm_port_stub: LLMPort,
    memory_port_stub: MemoryPort,
) -> None:
    """
    Slow cancelled-run script generation emits a bounded heartbeat before the script event.
    """

    harness = await _run_harness(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        llm_port_stub=llm_port_stub,
        memory_port_stub=memory_port_stub,
        executor=CancelledExecutor,
        script_delay=0.6,
        script_result="tap done",
        configuration=cancellation_configuration(script_timeout=1.0, heartbeat_threshold=0.5),
    )

    heartbeat_event = _history_script_heartbeat_events(harness)[0]
    event_types = [
        event.get("type")
        for event in harness.telemetry.events
        if event.get("type")
        in {
            FathomEvent.WORKFLOW_CANCELLED,
            FathomEvent.PHASE_HEARTBEAT,
            FathomEvent.SCRIPT_GENERATED,
        }
        and event.get("phase") in {None, FinalizationPhase.HISTORY_SCRIPT.value}
    ]

    assert event_types.index(FathomEvent.WORKFLOW_CANCELLED) < event_types.index(
        FathomEvent.PHASE_HEARTBEAT
    )
    assert event_types.index(FathomEvent.PHASE_HEARTBEAT) < event_types.index(
        FathomEvent.SCRIPT_GENERATED
    )
    assert heartbeat_event["message"] == "Finalizing the script..."
    assert heartbeat_event["run_outcome"] == RunOutcome.CANCELLED.value
    assert heartbeat_event["step"] == 0
    assert heartbeat_event["workflow_id"] == "workflow-cancelled-script"


@pytest.mark.asyncio
async def test_cancelled_run_heartbeat_failure_does_not_flip_outcome(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    llm_port_stub: LLMPort,
    memory_port_stub: MemoryPort,
) -> None:
    """
    Heartbeat telemetry failures must not convert a cancelled run into a failed run.
    """

    import fathom.runtime.executor as executor_module

    harness = build_intent_strategy(
        tmp_path=tmp_path,
        llm=llm_port_stub,
        memory=memory_port_stub,
        configuration=cancellation_configuration(script_timeout=1.0, heartbeat_threshold=0.5),
    )
    original_info = harness.telemetry.info

    async def fail_only_heartbeat(message: str, **context: Any) -> None:
        """
        Raise only for the finalization heartbeat event.
        """

        if context.get("type") == FathomEvent.PHASE_HEARTBEAT:
            raise RuntimeError("telemetry heartbeat broke")

        await original_info(message, **context)

    async def get_current_script(
        self: HistoryService,
        *,
        intent: str,
        step_number: int,
    ) -> str:
        """
        Run long enough for the heartbeat task to attempt an emit.
        """

        _ = (self, intent, step_number)
        await asyncio.sleep(0.6)
        return "tap done"

    monkeypatch.setattr(executor_module, "GraphExecutor", CancelledExecutor)
    monkeypatch.setattr(HistoryService, "get_current_script", get_current_script)
    monkeypatch.setattr(harness.telemetry, "info", fail_only_heartbeat)

    result = await harness.strategy.execute()
    script_event = harness.telemetry.of_type(FathomEvent.SCRIPT_GENERATED)[0]

    assert result.is_cancelled is True
    assert script_event["message"] == "tap done"
    assert script_event["run_outcome"] == RunOutcome.CANCELLED.value
    assert any(
        "cancelled-run script heartbeat emit failed" in record.message for record in caplog.records
    )


@pytest.mark.asyncio
async def test_completed_run_script_exception_fails_without_empty_script(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    llm_port_stub: LLMPort,
    memory_port_stub: MemoryPort,
) -> None:
    """
    Completed-run script exceptions remain strict and do not emit empty success artefacts.
    """

    harness = await _run_harness(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        llm_port_stub=llm_port_stub,
        memory_port_stub=memory_port_stub,
        executor=CompletedExecutor,
        script_error=RuntimeError("exporter broke"),
    )

    script_events = harness.telemetry.of_type(FathomEvent.SCRIPT_GENERATED)

    assert script_events == []


@pytest.mark.asyncio
async def test_cancelled_run_propagates_task_cancellation_during_script_generation(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    llm_port_stub: LLMPort,
    memory_port_stub: MemoryPort,
) -> None:
    """
    Host task cancellation during script generation must propagate and must not emit fallback script.
    """

    import fathom.runtime.executor as executor_module

    script_started = asyncio.Event()
    harness = build_intent_strategy(
        tmp_path=tmp_path,
        llm=llm_port_stub,
        memory=memory_port_stub,
        configuration=cancellation_configuration(script_timeout=1.0, heartbeat_threshold=0.5),
    )

    async def get_current_script(
        self: HistoryService,
        *,
        intent: str,
        step_number: int,
    ) -> str:
        """
        Block until the caller cancels the strategy task.
        """

        _ = (self, intent, step_number)
        script_started.set()
        wait_forever: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        return await wait_forever

    monkeypatch.setattr(executor_module, "GraphExecutor", CancelledExecutor)
    monkeypatch.setattr(HistoryService, "get_current_script", get_current_script)

    task = asyncio.create_task(harness.strategy.execute())
    await asyncio.wait_for(script_started.wait(), timeout=1.0)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert harness.telemetry.of_type(FathomEvent.SCRIPT_GENERATED) == []
