from __future__ import annotations

import asyncio
import logging
from typing import Optional

from fathom.adapters.icon.noop import NoopIconDetector
from fathom.adapters.journal.noop import NoopRuntimeJournal
from fathom.adapters.ocr.noop import NoopOcr
from fathom.adapters.perception.overlay.noop import NoopOverlayDetector
from fathom.base.paths import SharedPathManager
from fathom.core.agent.planner import StepPlanner
from fathom.core.agent.reasoner import Reasoner
from fathom.core.agent.state import AgentState
from fathom.core.artifact.pipeline import ArtifactPipeline
from fathom.core.context.manager import ContextManager
from fathom.core.localization import EnsembleLocalizerService
from fathom.core.perception.localization import TargetLocalizationService
from fathom.core.perception.observation import ScreenObservationService
from fathom.core.runtime import RuntimeEventEmitter
from fathom.core.services.action import ActionExecutor
from fathom.core.services.audit import AuditService
from fathom.core.services.comparator import ScreenComparator
from fathom.core.services.exporter import ScriptExporter
from fathom.core.services.hierarchy import HierarchyService
from fathom.core.services.history import HistoryService
from fathom.core.services.hitl import HITLService
from fathom.core.services.perception import PerceptionService
from fathom.core.services.resolution import ReferenceResolutionService
from fathom.core.services.trace import TraceService
from fathom.core.services.vision import VisionService
from fathom.interfaces.device import DevicePort
from fathom.interfaces.icon import IconDetectorPort
from fathom.interfaces.journal import RuntimeJournalPort
from fathom.interfaces.knowledge import KnowledgePort
from fathom.interfaces.llm import LLMPort
from fathom.interfaces.memory import MemoryPort
from fathom.interfaces.ocr import OcrPort
from fathom.interfaces.overlay import OverlayDetectorPort
from fathom.interfaces.perception import PerceptionPort
from fathom.interfaces.signal import SignalPort
from fathom.interfaces.storage import StoragePort
from fathom.interfaces.summarization import SummarizationPort
from fathom.interfaces.telemetry import TelemetryPort
from fathom.processing.parsers.signature import HierarchySignatureBuilder
from fathom.schemas.configuration import FathomConfiguration
from fathom.schemas.exploration import ExplorationGraph
from fathom.schemas.metrics import ExecutionMetrics
from fathom.schemas.perception import PerceptionConfiguration  # noqa: TC001
from fathom.schemas.run import RealignmentPolicy

logger = logging.getLogger(__name__)


class GraphContext:
    """
    Context container for graph nodes.

    Holds ports, services, and mutable agent state.
    """

    def __init__(
        self,
        *,
        intent: str,
        llm: LLMPort,
        device: DevicePort,
        memory: MemoryPort,
        signal: SignalPort,
        storage: StoragePort,
        telemetry: TelemetryPort,
        perception: PerceptionPort,
        path_manager: SharedPathManager,
        configuration: FathomConfiguration,
        perception_configuration: PerceptionConfiguration,
        use_xml: bool,
        max_steps: int,
        workflow_id: str,
        package_name: str,
        ocr: Optional[OcrPort] = None,
        reasoner: Optional[Reasoner] = None,
        trace: Optional[TraceService] = None,
        planner: Optional[StepPlanner] = None,
        auditor: Optional[AuditService] = None,
        vision: Optional[VisionService] = None,
        icons: Optional[IconDetectorPort] = None,
        agent_state: Optional[AgentState] = None,
        history: Optional[HistoryService] = None,
        knowledge: Optional[KnowledgePort] = None,
        metrics: Optional[ExecutionMetrics] = None,
        cancel_event: Optional[asyncio.Event] = None,
        hierarchy: Optional[HierarchyService] = None,
        journal: Optional[RuntimeJournalPort] = None,
        comparator: Optional[ScreenComparator] = None,
        summarizer: Optional[SummarizationPort] = None,
        realignment: Optional[RealignmentPolicy] = None,
        context_manager: Optional[ContextManager] = None,
        action_executor: Optional[ActionExecutor] = None,
        pixel_overlay: Optional[OverlayDetectorPort] = None,
        ensemble: Optional[EnsembleLocalizerService] = None,
        exploration_graph: Optional[ExplorationGraph] = None,
        perception_service: Optional[PerceptionService] = None,
        resolution: Optional[ReferenceResolutionService] = None,
        screen_observer: Optional[ScreenObservationService] = None,
        target_localizer: Optional[TargetLocalizationService] = None,
        artifact_pipeline: Optional[ArtifactPipeline] = None,
    ) -> None:
        self.__intent = intent
        self.__device = device
        self.__perception_port = perception

        self.__llm = llm
        self.__memory = memory
        self.__storage = storage
        self.__knowledge = knowledge

        self.__telemetry = telemetry
        self.__path_manager = path_manager

        self.__exploration_graph = exploration_graph or ExplorationGraph()

        self.__use_xml = use_xml
        self.__max_steps = max_steps
        self.__workflow_id = workflow_id
        self.__package_name = package_name
        self.__configuration = configuration
        self.__perception_configuration = perception_configuration

        self.__cancel_event = cancel_event or asyncio.Event()
        self.__realignment = realignment or RealignmentPolicy()

        # Injected services with defaults for backward compatibility
        self.__metrics = metrics or ExecutionMetrics()

        self.__reasoner = reasoner or Reasoner(intent=intent)
        self.__agent_state = agent_state or AgentState(
            intent=intent,
            max_steps=max_steps,
            realignment_budget=self.__realignment.budget,
        )

        self.__signal = signal
        self.__hitl = HITLService(signal=signal, telemetry=telemetry)

        self.__artifact_pipeline = artifact_pipeline

        self.__perception = perception_service or PerceptionService(
            storage=storage,
            perception=perception,
            session_id=workflow_id,
            hierarchy_signature_builder=HierarchySignatureBuilder(),
            pipeline=artifact_pipeline,
        )

        # GCC Context Manager with optional summarizer
        self.__context_manager = context_manager or ContextManager(
            memory=memory, workflow_id=workflow_id, summarizer=summarizer
        )

        self.__auditor = auditor or AuditService()

        self.__vision = vision or VisionService(
            llm=llm,
            memory=memory,
            telemetry=telemetry,
            session_id=workflow_id,
            auditor=self.__auditor,
            use_cache=configuration.llm.use_cache,
        )

        self.__action_executor = action_executor or ActionExecutor(
            device=device,
            storage=storage,
            telemetry=telemetry,
            path_manager=path_manager,
            pipeline=artifact_pipeline,
        )

        self.__comparator = comparator or ScreenComparator()
        self.__hierarchy = hierarchy or HierarchyService(
            storage=storage,
            configuration=perception_configuration,
            pipeline=artifact_pipeline,
        )
        self.__planner = planner or StepPlanner(vision_tool=self.__vision)

        self.__history = history or HistoryService(
            storage=storage,
            workflow_id=workflow_id,
            package_name=package_name,
            path_manager=path_manager,
            exporter=ScriptExporter(llm=llm, use_cache=configuration.llm.use_cache),
            pipeline=artifact_pipeline,
        )
        self.__trace = trace or TraceService(path_manager=path_manager)
        self.__resolution = resolution or ReferenceResolutionService(
            ledger=memory,
            workflow_id=workflow_id,
        )

        self.__ocr = ocr or NoopOcr()
        self.__icons = icons or NoopIconDetector()
        self.__pixel_overlay = pixel_overlay or NoopOverlayDetector()
        self.__ensemble = ensemble or EnsembleLocalizerService(workflow_id=workflow_id)

        self.__screen_observer = screen_observer or ScreenObservationService(
            configuration=perception_configuration,
            ocr=self.__ocr,
            icons=self.__icons,
            device=device,
            workflow_id=workflow_id,
            pixel_overlay=self.__pixel_overlay,
            pipeline=artifact_pipeline,
        )
        self.__target_localizer = target_localizer or TargetLocalizationService(
            workflow_id=workflow_id,
            ensemble=self.__ensemble,
        )

        self.__journal = journal if journal is not None else NoopRuntimeJournal()

        self.__event_emitter = RuntimeEventEmitter(
            journal=self.__journal,
            workflow_id=workflow_id,
        )

    @property
    def intent(self) -> str:
        """
        Returns the original intent.
        """

        return self.__intent

    @property
    def device(self) -> DevicePort:
        """
        Returns the DevicePort instance.
        """

        return self.__device

    @property
    def perception_port(self) -> PerceptionPort:
        """
        Returns the PerceptionPort instance.
        """

        return self.__perception_port

    @property
    def llm(self) -> LLMPort:
        """
        Returns the LLMPort instance.
        """

        return self.__llm

    @property
    def memory(self) -> MemoryPort:
        """
        Returns the MemoryPort instance.
        """

        return self.__memory

    @property
    def knowledge(self) -> Optional[KnowledgePort]:
        """
        Returns the KnowledgePort instance.
        """

        return self.__knowledge

    @property
    def storage(self) -> StoragePort:
        """
        Returns the StoragePort instance.
        """

        return self.__storage

    @property
    def telemetry(self) -> TelemetryPort:
        """
        Returns the TelemetryPort instance.
        """

        return self.__telemetry

    @property
    def comparator(self) -> ScreenComparator:
        """
        Returns the screen comparator service.
        """

        return self.__comparator

    @property
    def signal(self) -> SignalPort:
        """
        Returns the SignalPort instance.

        Note: For HITL operations with event emission, use context.hitl instead.
        This property is kept for backward compatibility and type checking.
        """

        return self.__signal

    @property
    def hitl(self) -> HITLService:
        """
        Returns the HITLService instance.
        """

        return self.__hitl

    @property
    def path_manager(self) -> SharedPathManager:
        """
        Returns the SharedPathManager instance.
        """

        return self.__path_manager

    @property
    def max_steps(self) -> int:
        """
        Returns the maximum number of steps allowed.
        """

        return self.__max_steps

    @property
    def use_xml(self) -> bool:
        """
        Boolean value indicating whether to use XML in grounding.
        """

        return self.__use_xml

    @property
    def workflow_id(self) -> str:
        """
        Returns the workflow ID.
        """

        return self.__workflow_id

    @property
    def package_name(self) -> str:
        """
        Returns the set package name.
        """

        return self.__package_name

    @property
    def realignment(self) -> RealignmentPolicy:
        """
        Returns the RealignmentPolicy instance.
        """

        return self.__realignment

    @property
    def configuration(self) -> FathomConfiguration:
        """
        Returns the FathomConfiguration instance.
        """

        return self.__configuration

    @property
    def is_cancelled(self) -> bool:
        """
        Boolean value indicating whether the context is cancelled.
        """

        return self.__cancel_event.is_set()

    def cancel(self) -> None:
        """
        Sets the cancellation event to stop execution.
        """

        self.__cancel_event.set()

    @property
    def exploration_graph(self) -> ExplorationGraph:
        """
        Returns the ExplorationGraph instance.

        Note: ExplorationGraph is designed to be mutated.
        If immutability is needed, implement copy() method on ExplorationGraph.
        """

        return self.__exploration_graph

    @property
    def agent_state(self) -> AgentState:
        """
        Returns the AgentState instance.
        Note: AgentState is intentionally mutable as it tracks execution progress.
        """

        return self.__agent_state

    def set_agent_state(self, state: AgentState) -> None:
        """
        Set/replace the AgentState instance (used for checkpoint restore).
        """

        self.__agent_state = state

    @property
    def context_manager(self) -> ContextManager:
        """
        Returns the ContextManager instance.
        """

        return self.__context_manager

    @property
    def action_executor(self) -> ActionExecutor:
        """
        Returns the ActionExecutor instance.
        """

        return self.__action_executor

    @property
    def reasoner(self) -> Reasoner:
        """
        Returns the Reasoner instance.
        """

        return self.__reasoner

    @property
    def metrics(self) -> ExecutionMetrics:
        """
        Returns the ExecutionMetrics instance.
        Note: Metrics are intentionally mutable for recording execution data.
        """

        return self.__metrics

    @property
    def vision(self) -> VisionService:
        """
        Returns the VisionService instance.
        """

        return self.__vision

    @property
    def planner(self) -> StepPlanner:
        """
        Returns the StepPlanner instance.
        """

        return self.__planner

    @property
    def auditor(self) -> AuditService:
        """
        Returns the AuditService instance for console logging.
        """

        return self.__auditor

    @property
    def hierarchy(self) -> HierarchyService:
        """
        Returns the HierarchyService instance.
        """

        return self.__hierarchy

    @property
    def history(self) -> HistoryService:
        """
        Returns the HistoryService instance.
        """

        return self.__history

    @property
    def trace(self) -> TraceService:
        """
        Returns the TraceService instance.
        """

        return self.__trace

    @property
    def perception(self) -> PerceptionService:
        """
        Returns the PerceptionService instance.
        """

        return self.__perception

    @property
    def resolution(self) -> ReferenceResolutionService:
        """
        Returns the ReferenceResolutionService instance.
        """

        return self.__resolution

    @property
    def ocr(self) -> OcrPort:
        """
        Returns the injected OCR port used by perception and localization.
        """

        return self.__ocr

    @property
    def icons(self) -> IconDetectorPort:
        """
        Returns the injected icon detector used by perception.
        """

        return self.__icons

    @property
    def pixel_overlay(self) -> OverlayDetectorPort:
        """
        Returns the injected pixel-overlay detector used by perception.
        """

        return self.__pixel_overlay

    @property
    def ensemble(self) -> EnsembleLocalizerService:
        """
        Returns the ensemble localizer service used by target localization.
        """

        return self.__ensemble

    @property
    def screen_observer(self) -> ScreenObservationService:
        """
        Returns the screen observation service.
        """

        return self.__screen_observer

    @property
    def target_localizer(self) -> TargetLocalizationService:
        """
        Returns the target localization service.
        """

        return self.__target_localizer

    def journal(self) -> RuntimeJournalPort:
        """
        Returns the injected runtime journal port.
        """

        return self.__journal

    @property
    def event_emitter(self) -> RuntimeEventEmitter:
        """
        Returns the runtime event emitter wired against the journal port.
        """

        return self.__event_emitter

    @property
    def artifact_pipeline(self) -> Optional[ArtifactPipeline]:
        """
        Return the artifact pipeline producers emit into, or ``None`` when disabled.
        """

        return self.__artifact_pipeline

    async def shutdown(self) -> None:
        """
        Drain background tasks from all owned services before teardown.
        """

        for service in (self.__action_executor, self.__hierarchy, self.__history):
            if hasattr(service, "drain_background_tasks"):
                try:
                    await service.drain_background_tasks()
                except Exception as exception:
                    logger.warning(
                        f"[graph-context] drain failed for {type(service).__name__}: {exception}"
                    )

        if self.__artifact_pipeline is not None:
            try:
                await self.__artifact_pipeline.drain()
            except Exception as exception:
                logger.warning(f"[graph-context] artifact pipeline drain failed: {exception}")
