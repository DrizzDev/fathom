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

from fathom.adapters.evidence.history import HistoryEvidenceSource
from fathom.constants.authoring import AuthoringArtifactKind, AuthoringStatus
from fathom.constants.dialect import DialectName
from fathom.constants.events import FathomEvent
from fathom.constants.generation import ScriptSource, ScriptStatus
from fathom.constants.state import RunOutcome
from fathom.core.services.decomposer import IntentDecomposer
from fathom.core.services.history import HistoryService
from fathom.interfaces.authoring import AuthoringPort
from fathom.interfaces.llm import LLMPort
from fathom.interfaces.memory import MemoryPort
from fathom.schemas.authoring import AuthoringArtifact, AuthoringResponse, AuthoringTask
from fathom.schemas.flow import Evidence, RunObjective
from fathom.schemas.generation import BaselineArtifact, ScriptFileMetadata
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
            if event.get("phase") == "fathom.finalization.history.script"
        ]

    @staticmethod
    async def __baseline(
        *,
        history: HistoryService,
        step_number: int,
    ) -> BaselineArtifact:
        """
        Return the baseline script available at cancellation time.
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
        Return the quality script generated after cancellation.
        """

        _ = (intent, step_number)
        return "quality tap continue"

    @staticmethod
    async def __slow_quality(
        *,
        intent: str,
        step_number: int,
    ) -> str:
        """
        Finish slowly enough to prove heartbeat delivery before SCRIPT_GENERATED.
        """

        _ = (intent, step_number)
        await asyncio.sleep(0.6)
        return "quality tap continue"

    def __patch_boundaries(
        self,
        *,
        monkeypatch: pytest.MonkeyPatch,
        graph: TerminalIntentGraph,
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
            execution_id: str,
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
                artifacts=(execution_id,),
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

    async def test_cancelled_strategy_emits_heartbeat_then_script(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        llm_port_stub: LLMPort,
        memory_port_stub: MemoryPort,
    ) -> None:
        """
        Cancelled strategy finalization emits heartbeat while generating the quality script.
        """

        self.__patch_boundaries(
            monkeypatch=monkeypatch,
            graph=TerminalIntentGraph.workflow_cancelled(),
            baseline=self.__baseline,
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

        result = await harness.strategy.execute()

        stream = [
            event
            for event in harness.telemetry.events
            if event.get("type")
            in {
                FathomEvent.PHASE_HEARTBEAT,
                FathomEvent.SCRIPT_GENERATED,
            }
            and event.get("phase") in {None, "fathom.finalization.history.script"}
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

        assert script_event["message"] == "quality tap continue"
        assert script_event["source"] == ScriptSource.QUALITY.value
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
            baseline=self.__baseline,
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

        script_event = harness.telemetry.of_type(FathomEvent.SCRIPT_GENERATED)[0]

        assert script_event["message"] == "quality tap continue"
        assert script_event["source"] == ScriptSource.QUALITY.value
        assert script_event["run_outcome"] == RunOutcome.CANCELLED.value
        assert script_event["workflow_id"] == "workflow-cancelled-script"
