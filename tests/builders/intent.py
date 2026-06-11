from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from fathom.adapters.checkpoint import SqliteCheckpointStore
from fathom.adapters.signal.noop import NoopSignal
from fathom.adapters.storage.local import LocalStorage
from fathom.base.paths import SharedPathManager
from fathom.constants.events import FathomEvent
from fathom.constants.interaction import SwipeSpeed
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


class DeviceBoundary(DevicePort):
    """
    Device boundary double used because real adapters require a live device/session.
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

        return b"fake"

    async def dump_hierarchy(self) -> Optional[str]:
        """
        Return no hierarchy.
        """

        return None

    async def get_snapshot(self) -> Tuple[bytes, Optional[str]]:
        """
        Return placeholder screenshot and no hierarchy.
        """

        return b"fake", None

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


class PerceptionBoundary(PerceptionPort):
    """
    Perception boundary double used because real perception captures a live screen.
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

        return ScreenCapture(width=100, height=200, activity="Main", image=b"fake", timestamp=0)


class SummarizerBoundary(SummarizationPort):
    """
    Summarizer boundary double for constructor completeness.
    """

    async def summarize_trace(self, trace: List[Dict[str, Any]]) -> str:
        """
        Return a fixed trace summary.
        """

        _ = trace
        return "summary"


class DecomposerBoundary:
    """
    Decomposer boundary that returns deterministic sub-goals without an LLM call.
    """

    async def decompose(self, *, intent: str) -> list[Any]:
        """
        Return no sub-goals.
        """

        _ = intent
        return []


class TerminalGraph:
    """
    Terminal graph boundary for finalization tests.
    """

    async def aget_state(self, config: Any) -> Any:
        """
        Return a terminal graph snapshot.
        """

        from types import SimpleNamespace

        _ = config
        return SimpleNamespace(next=(), values={})


class CancelledExecutor:
    """
    Executor boundary that emits the existing cancellation acknowledgement and cancels context.
    """

    def __init__(self, *, context: Any, graph: Any, thread_id: str, **_: Any) -> None:
        """
        Capture the graph context passed by IntentStrategy.
        """

        self.__context = context
        self.__graph = graph
        self.__thread_id = thread_id

    async def run(self) -> None:
        """
        Simulate cooperative cancellation from the executor layer.
        """

        _ = (self.__graph, self.__thread_id)
        self.__context.cancel()
        await self.__context.telemetry.info(
            "Stopping the run.",
            type=FathomEvent.WORKFLOW_CANCELLED,
        )


class CompletedExecutor:
    """
    Executor boundary that finishes without cancelling context.
    """

    def __init__(self, **_: Any) -> None:
        """
        Accept GraphExecutor constructor arguments.
        """

    async def run(self) -> None:
        """
        Complete normally.
        """


class IntentHarness:
    """
    Intent strategy plus recording telemetry for one test run.
    """

    def __init__(self, *, strategy: IntentStrategy, telemetry: RecordingTelemetry) -> None:
        """
        Bind strategy and telemetry.
        """

        self.strategy = strategy
        self.telemetry = telemetry


def build_intent_strategy(
    *,
    tmp_path: Path,
    llm: LLMPort,
    memory: MemoryPort,
    configuration: FathomConfiguration,
) -> IntentHarness:
    """
    Build IntentStrategy with real safe adapters and controlled external boundaries.
    """

    telemetry = RecordingTelemetry()
    path_manager = SharedPathManager(settings=FathomSettings(assets_path=tmp_path / "assets"))
    strategy = IntentStrategy(
        intent="Cancel after collecting a partial script",
        llm=llm,
        device=DeviceBoundary(),
        memory=memory,
        signal=NoopSignal(),
        storage=LocalStorage(path_manager=path_manager),
        telemetry=telemetry,
        perception=PerceptionBoundary(),
        summarizer=SummarizerBoundary(),
        configuration=configuration,
        use_xml=False,
        max_steps=5,
        package_name="com.example",
        workflow_id="workflow-cancelled-script",
        path_manager=path_manager,
        checkpoint_store=SqliteCheckpointStore(
            directory=path_manager.get_checkpoint_directory(),
            policy=SqliteCheckpointPolicy(),
        ),
    )
    return IntentHarness(strategy=strategy, telemetry=telemetry)


def cancellation_configuration(
    *,
    script_timeout: float = 1.0,
    heartbeat_threshold: float = 0.5,
) -> FathomConfiguration:
    """
    Build the short-budget configuration used by cancellation-finalization tests.
    """

    finalization = FinalizationBudgetPolicy(
        history=HistoryFinalizationBudget(flush=10.0, script=script_timeout),
        graph=GraphFinalizationBudget(state_read=1.0),
        runtime=RuntimeFinalizationBudget(
            cleanup=1.0,
            context_shutdown=1.0,
            background_drain=1.0,
            memory_summary=1.0,
        ),
    )
    return FathomConfiguration(
        intent=IntentConfiguration(finalization=finalization),
        telemetry=TelemetryConfiguration(
            phase=PhaseMessage(
                heartbeat=HeartbeatBudget(
                    threshold=heartbeat_threshold,
                    limit=5,
                    message="Still working...",
                    script_finalization="Finalizing the script...",
                )
            )
        ),
    )
