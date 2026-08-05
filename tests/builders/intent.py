from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, List, Optional, Tuple

from fathom.adapters.checkpoint import LangGraphPlanStore, SqliteCheckpointStore
from fathom.adapters.signal.noop import NoopSignal
from fathom.adapters.storage.local import LocalStorage
from fathom.base.paths import SharedPathManager
from fathom.constants.events import FathomEvent
from fathom.constants.interaction import SwipeSpeed
from fathom.core.capability.catalog import CommandCatalogProvider
from fathom.core.exceptions import WorkflowCancelledError
from fathom.interfaces.authoring import AuthoringPort
from fathom.interfaces.device import DevicePort
from fathom.interfaces.llm import LLMPort
from fathom.interfaces.memory import MemoryPort
from fathom.interfaces.perception import PerceptionPort
from fathom.interfaces.summarization import SummarizationPort
from fathom.interfaces.telemetry import TelemetryPort
from fathom.schemas.checkpoint import SqliteCheckpointPolicy
from fathom.schemas.configuration import (
    DeviceRuntimeConfiguration,
    FathomConfiguration,
    IntentConfiguration,
    TelemetryConfiguration,
)
from fathom.schemas.finalization import (
    FinalizationBudgetPolicy,
    GraphFinalizationBudget,
    HistoryFinalizationBudget,
    RuntimeFinalizationBudget,
)
from fathom.schemas.results import ActionResult
from fathom.schemas.screens import ScreenCapture
from fathom.schemas.subgoal import SubGoal
from fathom.schemas.success import ObservationRequirement, ObservedSuccess
from fathom.schemas.telemetry import HeartbeatBudget, PhaseMessage
from fathom.settings.env import FathomSettings
from fathom.strategies.intent import IntentStrategy

if TYPE_CHECKING:
    from pathlib import Path


class RecordingTelemetry(TelemetryPort):
    """
    Telemetry port that records emitted client events for assertions.
    """

    def __init__(self) -> None:
        """
        Initialize the in-memory event sink.
        """

        self.events: List[Dict[str, Any]] = []

    async def debug(self, message: str, **context: Any) -> None:
        """
        Record a debug event.
        """

        self.__record(level="debug", message=message, context=context)

    async def info(self, message: str, **context: Any) -> None:
        """
        Record an info event.
        """

        self.__record(level="info", message=message, context=context)

    async def warning(self, message: str, **context: Any) -> None:
        """
        Record a warning event.
        """

        self.__record(level="warning", message=message, context=context)

    async def error(self, message: str, **context: Any) -> None:
        """
        Record an error event.
        """

        self.__record(level="error", message=message, context=context)

    async def exception(
        self,
        message: str,
        *,
        exception: Optional[BaseException] = None,
        **context: Any,
    ) -> None:
        """
        Record an exception event.
        """

        if exception is not None:
            context["exception"] = exception
        self.__record(level="exception", message=message, context=context)

    def of_type(self, event_type: FathomEvent) -> List[Dict[str, Any]]:
        """
        Return all events with the requested Fathom event type.
        """

        return [event for event in self.events if event.get("type") == event_type]

    def __record(self, *, level: str, message: str, context: Dict[str, Any]) -> None:
        """
        Append one normalized telemetry record.
        """

        self.events.append({"level": level, "message": message, **context})


class DeterministicDevicePort(DevicePort):
    """
    Deterministic device port for tests that cannot require a live device session.
    """

    @property
    def configuration(self) -> Optional[DeviceRuntimeConfiguration]:
        """
        Return deterministic device configuration.
        """

        return DeviceRuntimeConfiguration(identifier="test-device")

    async def tap(self, *, x: int, y: int) -> ActionResult:
        """
        Return a successful no-op tap result.
        """

        _ = (x, y)
        return ActionResult(success=True, duration=0)

    async def type(
        self,
        *,
        text: str,
        prefilled: str = "",
        replace: bool = True,
        locator: Optional[str] = None,
    ) -> ActionResult:
        """
        Return a successful no-op type result.
        """

        _ = (text, prefilled, replace, locator)
        return ActionResult(success=True, duration=0)

    async def swipe(
        self,
        *,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        duration: Optional[int] = None,
        speed: Optional[SwipeSpeed] = None,
    ) -> ActionResult:
        """
        Return a successful no-op swipe result.
        """

        _ = (x1, y1, x2, y2, duration, speed)
        return ActionResult(success=True, duration=0)

    async def back(self) -> ActionResult:
        """
        Return a successful no-op back result.
        """

        return ActionResult(success=True, duration=0)

    async def home(self) -> ActionResult:
        """
        Return a successful no-op home result.
        """

        return ActionResult(success=True, duration=0)

    async def get_current_package(self) -> str:
        """
        Return deterministic package name.
        """

        return "com.example"

    async def capture_screen(self) -> bytes:
        """
        Return placeholder screenshot bytes.
        """

        return b"placeholder-screen"

    async def dump_hierarchy(self) -> Optional[str]:
        """
        Return no hierarchy.
        """

        return None

    async def get_snapshot(self) -> Tuple[bytes, Optional[str]]:
        """
        Return placeholder screenshot and no hierarchy.
        """

        return b"placeholder-screen", None

    async def get_dimensions(self) -> Tuple[int, int]:
        """
        Return deterministic screen dimensions.
        """

        return 100, 200

    async def wait_for_device(self, *, timeout: float) -> bool:
        """
        Return ready immediately.
        """

        _ = timeout
        return True


class DeterministicPerceptionPort(PerceptionPort):
    """
    Deterministic perception port for tests that cannot capture a live screen.
    """

    @property
    def configuration(self) -> Optional[DeviceRuntimeConfiguration]:
        """
        Return deterministic device configuration.
        """

        return DeviceRuntimeConfiguration(identifier="test-device")

    async def capture(self) -> ScreenCapture:
        """
        Return a placeholder screen capture.
        """

        return ScreenCapture(
            width=100,
            height=200,
            timestamp=0,
            activity="Main",
            image=b"placeholder-screen",
        )


class DeterministicSummarizationPort(SummarizationPort):
    """
    Deterministic summarization port for constructor-complete strategy tests.
    """

    async def summarize_trace(self, trace: List[Dict[str, Any]]) -> str:
        """
        Return a fixed trace summary.
        """

        _ = trace
        return "summary"


class DeterministicDecomposer:
    """
    Deterministic decomposer that avoids an LLM call.
    """

    async def decompose(self, *, intent: str) -> List[SubGoal]:
        """
        Return a single deterministic sub-goal so an accepted plan always has one goal.
        """

        return [SubGoal(index=0, objective=intent, success=ObservedSuccess(observation=ObservationRequirement(assertion=intent)))]


class TerminalIntentGraph:
    """
    Deterministic graph stream that exercises the real GraphExecutor lifecycle.
    """

    def __init__(self, *, stream_exception: Optional[BaseException] = None) -> None:
        """
        Initialize the graph stream outcome.
        """

        self.__stream_exception = stream_exception

    @classmethod
    def completed(cls) -> TerminalIntentGraph:
        """
        Build a graph that streams to completion.
        """

        return cls()

    @classmethod
    def workflow_cancelled(cls) -> TerminalIntentGraph:
        """
        Build a graph that surfaces cooperative workflow cancellation.
        """

        return cls(stream_exception=WorkflowCancelledError(workflow_id="workflow-cancelled-script"))

    @classmethod
    def host_cancelled(cls) -> TerminalIntentGraph:
        """
        Build a graph that surfaces host task cancellation.
        """

        return cls(stream_exception=asyncio.CancelledError())

    async def astream(self, input_value: Any, *, config: Any) -> AsyncIterator[Dict[str, Any]]:
        """
        Stream a deterministic terminal graph outcome.
        """

        _ = (input_value, config)

        if self.__stream_exception is not None:
            raise self.__stream_exception

        if False:
            yield {}

    async def aget_state(self, config: Any) -> Any:
        """
        Return a terminal graph snapshot.
        """

        _ = config
        return SimpleNamespace(next=(), values={})

    async def aupdate_state(self, config: Any, values: Any, as_node: Optional[str] = None) -> Any:
        """
        Accept a pre-graph plan seed without persisting (deterministic double).
        """

        _ = (config, values, as_node)
        return config


class IntentStrategyHarness:
    """
    Intent strategy plus recording telemetry for one test run.
    """

    def __init__(self, *, strategy: IntentStrategy, telemetry: RecordingTelemetry) -> None:
        """
        Bind strategy and telemetry.
        """

        self.strategy = strategy
        self.telemetry = telemetry


class IntentStrategyHarnessBuilder:
    """
    Builds IntentStrategy with real application infrastructure and controlled external ports.
    """

    @staticmethod
    def build(
        *,
        llm: LLMPort,
        tmp_path: Path,
        memory: MemoryPort,
        configuration: FathomConfiguration,
        authoring: Optional[AuthoringPort] = None,
    ) -> IntentStrategyHarness:
        """
        Build an IntentStrategy harness for cancellation finalization tests.
        """

        telemetry = RecordingTelemetry()
        intent = "Cancel after collecting a partial script"
        path_manager = SharedPathManager(
            settings=FathomSettings.model_validate({"assets_path": tmp_path / "assets"})
        )

        strategy = IntentStrategy(
            llm=llm,
            max_steps=5,
            intent=intent,
            tenant="test-tenant",
            thread="test-thread",
            requester="test-requester",
            responder="test-responder",
            memory=memory,
            use_xml=False,
            telemetry=telemetry,
            signal=NoopSignal(),
            catalog=CommandCatalogProvider().build(),
            path_manager=path_manager,
            package_name="com.example",
            configuration=configuration,
            device=DeterministicDevicePort(),
            execution_id="execution-cancelled-script",
            workflow_id="workflow-cancelled-script",
            perception=DeterministicPerceptionPort(),
            summarizer=DeterministicSummarizationPort(),
            storage=LocalStorage(path_manager=path_manager),
            authoring=authoring,
            plans=LangGraphPlanStore(),
            checkpoint_store=SqliteCheckpointStore(
                policy=SqliteCheckpointPolicy(),
                directory=path_manager.get_checkpoint_directory(),
            ),
        )
        return IntentStrategyHarness(strategy=strategy, telemetry=telemetry)


class IntentCancellationConfigurationBuilder:
    """
    Builds cancellation-finalization configuration for IntentStrategy tests.
    """

    @staticmethod
    def build(
        *,
        script_timeout: float = 1.0,
        heartbeat_threshold: float = 0.5,
    ) -> FathomConfiguration:
        """
        Build the short-budget configuration used by cancellation-finalization tests.
        """

        finalization = FinalizationBudgetPolicy(
            graph=GraphFinalizationBudget(state_read=1.0),
            history=HistoryFinalizationBudget(flush=10.0, script=script_timeout),
            runtime=RuntimeFinalizationBudget(
                cleanup=1.0,
                memory_summary=1.0,
                context_shutdown=1.0,
                background_drain=1.0,
            ),
        )
        return FathomConfiguration(
            intent=IntentConfiguration(finalization=finalization),
            telemetry=TelemetryConfiguration(
                phase=PhaseMessage(
                    heartbeat=HeartbeatBudget(
                        limit=5,
                        message="Still working...",
                        threshold=heartbeat_threshold,
                        script_finalization="Finalizing the script...",
                    )
                )
            ),
        )
