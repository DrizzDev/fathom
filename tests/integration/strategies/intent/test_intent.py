from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Awaitable, Callable, Dict, List

import pytest
from tests.builders.intent import (
    DeterministicDecomposer,
    IntentCancellationConfigurationBuilder,
    IntentStrategyHarness,
    IntentStrategyHarnessBuilder,
    TerminalIntentGraph,
)

from fathom.constants.events import FathomEvent
from fathom.constants.finalization import FinalizationPhase
from fathom.constants.state import RunOutcome
from fathom.core.services.decomposer import IntentDecomposer
from fathom.core.services.history import HistoryService
from fathom.interfaces.llm import LLMPort
from fathom.interfaces.memory import MemoryPort
from fathom.strategies.graph.intent.builder import IntentGraphBuilder

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.asyncio
class TestIntentStrategyCancelledScriptDelivery:
    """
    Integration coverage for cancelled-run script delivery through IntentStrategy.
    """

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
        Return the partial script available at cancellation time.
        """

        _ = (history, intent, step_number)
        return "tap continue"

    @staticmethod
    async def __slow_script(
        *,
        history: HistoryService,
        intent: str,
        step_number: int,
    ) -> str:
        """
        Finish slowly enough to prove heartbeat delivery before SCRIPT_GENERATED.
        """

        _ = (history, intent, step_number)
        await asyncio.sleep(0.6)
        return "tap continue"

    def __patch_boundaries(
        self,
        *,
        monkeypatch: pytest.MonkeyPatch,
        graph: TerminalIntentGraph,
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

    async def test_cancelled_strategy_emits_heartbeat_then_script(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        llm_port_stub: LLMPort,
        memory_port_stub: MemoryPort,
    ) -> None:
        """
        Cancelled strategy finalization emits heartbeat and partial script.
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

        result = await harness.strategy.execute()

        stream = [
            event
            for event in harness.telemetry.events
            if event.get("type")
            in {
                FathomEvent.PHASE_HEARTBEAT,
                FathomEvent.SCRIPT_GENERATED,
            }
            and event.get("phase") in {None, FinalizationPhase.HISTORY_SCRIPT.value}
        ]
        event_types = [event["type"] for event in stream]
        heartbeat_event = self.__history_script_heartbeat_events(harness=harness)[0]
        script_event = harness.telemetry.of_type(FathomEvent.SCRIPT_GENERATED)[0]

        assert result.is_cancelled is True
        assert event_types == [
            FathomEvent.PHASE_HEARTBEAT,
            FathomEvent.SCRIPT_GENERATED,
        ]
        assert heartbeat_event["message"] == "Finalizing the script..."
        assert heartbeat_event["run_outcome"] == RunOutcome.CANCELLED.value
        assert heartbeat_event["workflow_id"] == "workflow-cancelled-script"

        assert script_event["message"] == "tap continue"
        assert script_event["run_outcome"] == RunOutcome.CANCELLED.value
        assert script_event["workflow_id"] == "workflow-cancelled-script"

    async def test_host_cancelled_strategy_emits_partial_script_before_cancellation_propagates(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        llm_port_stub: LLMPort,
        memory_port_stub: MemoryPort,
    ) -> None:
        """
        Host-level cancellation cannot bypass partial-script delivery.
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

        script_event = harness.telemetry.of_type(FathomEvent.SCRIPT_GENERATED)[0]

        assert script_event["message"] == "tap continue"
        assert script_event["run_outcome"] == RunOutcome.CANCELLED.value
        assert script_event["workflow_id"] == "workflow-cancelled-script"
