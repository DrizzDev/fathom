from __future__ import annotations

import asyncio
from typing import Any

import pytest
from tests.builders.intent import (
    CancelledExecutor,
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


@pytest.mark.asyncio
async def test_cancelled_strategy_emits_heartbeat_then_script_after_cancel_event(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    llm_port_stub: LLMPort,
    memory_port_stub: MemoryPort,
) -> None:
    """
    Integration proof for the client stream: cancel ack first, finalization heartbeat, then partial script.
    """

    import fathom.runtime.executor as executor_module

    def build_graph(
        self: IntentGraphBuilder,
        *,
        checkpointer: object,
        interrupt_before: list[str],
    ) -> TerminalGraph:
        """
        Return the terminal graph boundary.
        """

        _ = (self, checkpointer, interrupt_before)
        return TerminalGraph()

    async def get_current_script(
        self: HistoryService,
        *,
        intent: str,
        step_number: int,
    ) -> str:
        """
        Finish slowly enough to prove heartbeat delivery before SCRIPT_GENERATED.
        """

        _ = (self, intent, step_number)
        await asyncio.sleep(0.6)
        return "tap continue"

    monkeypatch.setattr(
        IntentDecomposer,
        "with_configuration",
        staticmethod(lambda **_: DecomposerBoundary()),
    )
    monkeypatch.setattr(IntentGraphBuilder, "build", build_graph)
    monkeypatch.setattr(executor_module, "GraphExecutor", CancelledExecutor)
    monkeypatch.setattr(HistoryService, "get_current_script", get_current_script)

    harness = build_intent_strategy(
        tmp_path=tmp_path,
        llm=llm_port_stub,
        memory=memory_port_stub,
        configuration=cancellation_configuration(script_timeout=1.0, heartbeat_threshold=0.5),
    )
    result = await harness.strategy.execute()

    stream = [
        event
        for event in harness.telemetry.events
        if event.get("type")
        in {
            FathomEvent.WORKFLOW_CANCELLED,
            FathomEvent.PHASE_HEARTBEAT,
            FathomEvent.SCRIPT_GENERATED,
        }
        and event.get("phase") in {None, FinalizationPhase.HISTORY_SCRIPT.value}
    ]
    event_types = [event["type"] for event in stream]
    heartbeat_event = _history_script_heartbeat_events(harness)[0]
    script_event = harness.telemetry.of_type(FathomEvent.SCRIPT_GENERATED)[0]

    assert result.is_cancelled is True
    assert event_types == [
        FathomEvent.WORKFLOW_CANCELLED,
        FathomEvent.PHASE_HEARTBEAT,
        FathomEvent.SCRIPT_GENERATED,
    ]
    assert heartbeat_event["message"] == "Finalizing the script..."
    assert heartbeat_event["run_outcome"] == RunOutcome.CANCELLED.value
    assert heartbeat_event["workflow_id"] == "workflow-cancelled-script"
    assert script_event["message"] == "tap continue"
    assert script_event["run_outcome"] == RunOutcome.CANCELLED.value
    assert script_event["workflow_id"] == "workflow-cancelled-script"
