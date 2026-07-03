from __future__ import annotations

import asyncio
import itertools
from logging import getLogger
from typing import FrozenSet, Optional, Sequence, Tuple

from fathom.adapters.dialect.drizz.factory import DrizzDialectFactory
from fathom.adapters.evidence.history import HistoryEvidenceSource
from fathom.adapters.icon.noop import NoopIconDetector
from fathom.adapters.journal.noop import NoopRuntimeJournal
from fathom.adapters.ocr.noop import NoopOcr
from fathom.adapters.perception.overlay.noop import NoopOverlayDetector
from fathom.adapters.script.refresher import BaselineRefresher
from fathom.authoring.agent import AuthoringAgent
from fathom.authoring.application.runner import AuthoringRunner
from fathom.authoring.evidence import AuthoringEvidenceBuilder
from fathom.base.paths import SharedPathManager
from fathom.constants.platform import DevicePlatform
from fathom.constants.tools import TurnMode
from fathom.core.agent.action import ActionBuilder
from fathom.core.agent.command import CommandGate
from fathom.core.agent.opener import OpenerSignalPolicy
from fathom.core.agent.planner import StepPlanner
from fathom.core.agent.reasoner import Reasoner
from fathom.core.agent.state import AgentState
from fathom.core.agent.tools import DEFAULT_TOOL_POLICIES, ToolScope
from fathom.core.artifact.pipeline import ArtifactPipeline
from fathom.core.capability.catalog import CommandCatalog, CommandCatalogProvider
from fathom.core.capture.store import CaptureStore
from fathom.core.context.manager import ContextManager
from fathom.core.dialect.policy import Policy
from fathom.core.embedding.cache import EmbeddingCache
from fathom.core.localization import EnsembleLocalizerService
from fathom.core.perception.localization import TargetLocalizationService
from fathom.core.perception.observation import ScreenObservationService
from fathom.core.prompts.generation import FlowPromptBuilder
from fathom.core.runtime import RuntimeEventEmitter
from fathom.core.services.abort import AbortDetectorFactory
from fathom.core.services.action import ActionExecutor
from fathom.core.services.artifacts import ArtifactCatalog
from fathom.core.services.audit import AuditService
from fathom.core.services.authoring import AuthoringService
from fathom.core.services.comparator import ScreenComparator
from fathom.core.services.generation.assembler import EvidenceAssembler
from fathom.core.services.generation.baseline import BaselineScriptService
from fathom.core.services.generation.binder import LaunchBinder
from fathom.core.services.generation.classifier import LauncherClassifier
from fathom.core.services.generation.distiller import Distiller
from fathom.core.services.generation.llm import LlmFlowGenerator
from fathom.core.services.generation.normalizer import RunTraceNormalizer
from fathom.core.services.generation.projector import DeterministicFlowGenerator
from fathom.core.services.generation.service import ScriptGenerationService
from fathom.core.services.hierarchy import HierarchyService
from fathom.core.services.history import HistoryService
from fathom.core.services.hitl import HITLService
from fathom.core.services.perception import PerceptionService
from fathom.core.services.recorder import ConversationRecorder
from fathom.core.services.resolution import ReferenceResolutionService
from fathom.core.services.telemetry import PhaseAnnouncer
from fathom.core.services.trace import TraceService
from fathom.core.services.vision import VisionService
from fathom.interfaces.abort import AbortDetectorPort
from fathom.interfaces.authoring import AuthoringPort
from fathom.interfaces.device import DevicePort
from fathom.interfaces.embedding import EmbeddingPort
from fathom.interfaces.evidence import EvidenceSource
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
from fathom.schemas.capabilities import (
    DeviceCapability,
    HITLCapability,
    RuntimeCapabilities,
)
from fathom.schemas.configuration import FathomConfiguration
from fathom.schemas.exploration import ExplorationGraph
from fathom.schemas.metrics import ExecutionMetrics
from fathom.schemas.perception import PerceptionConfiguration  # noqa: TC001
from fathom.schemas.run import RealignmentPolicy
from fathom.schemas.tools import ToolPolicyContext, ToolScopeMatrixExpansion

logger = getLogger(__name__)


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
        phase: PhaseAnnouncer,
        telemetry: TelemetryPort,
        perception: PerceptionPort,
        path_manager: SharedPathManager,
        configuration: FathomConfiguration,
        perception_configuration: PerceptionConfiguration,
        use_xml: bool,
        max_steps: int,
        execution_id: str,
        workflow_id: str,
        package_name: str,
        tenant: str,
        thread: str,
        requester: str,
        responder: str,
        workspace: Optional[str] = None,
        ocr: Optional[OcrPort] = None,
        reasoner: Optional[Reasoner] = None,
        trace: Optional[TraceService] = None,
        recorder: Optional[ConversationRecorder] = None,
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
        authoring: Optional[AuthoringPort] = None,
        pixel_overlay: Optional[OverlayDetectorPort] = None,
        ensemble: Optional[EnsembleLocalizerService] = None,
        exploration_graph: Optional[ExplorationGraph] = None,
        perception_service: Optional[PerceptionService] = None,
        resolution: Optional[ReferenceResolutionService] = None,
        screen_observer: Optional[ScreenObservationService] = None,
        target_localizer: Optional[TargetLocalizationService] = None,
        artifact_pipeline: Optional[ArtifactPipeline] = None,
        embedder: Optional[EmbeddingPort] = None,
        embedding_cache: Optional[EmbeddingCache] = None,
        abort_detector: Optional[AbortDetectorPort] = None,
    ) -> None:
        self.__intent = intent
        self.__device = device
        self.__perception_port = perception

        self.__embedder = embedder
        self.__embedding_cache: Optional[EmbeddingCache]

        if embedding_cache is not None:
            self.__embedding_cache = embedding_cache
        else:
            self.__embedding_cache = (
                EmbeddingCache(embedder=embedder) if embedder is not None else None
            )

        self.__llm = llm
        self.__memory = memory
        self.__storage = storage
        self.__knowledge = knowledge

        self.__phase = phase
        self.__telemetry = telemetry
        self.__path_manager = path_manager

        self.__exploration_graph = exploration_graph or ExplorationGraph()

        self.__use_xml = use_xml
        self.__max_steps = max_steps
        self.__execution_id = execution_id
        self.__workflow_id = workflow_id
        self.__tenant = tenant
        self.__thread = thread
        self.__workspace = workspace
        self.__requester = requester
        self.__responder = responder
        self.__package_name = package_name
        self.__configuration = configuration
        self.__perception_configuration = perception_configuration

        self.__cancel_event = cancel_event or asyncio.Event()
        self.__realignment = realignment or RealignmentPolicy()

        # Injected services with defaults for backward compatibility
        self.__metrics = metrics or ExecutionMetrics()

        self.__capabilities = RuntimeCapabilities(
            hitl=HITLCapability(enabled=signal.supports_interruption()),
            device=DeviceCapability(
                system_back_supported=self.__resolve_supports_back(device=device),
            ),
        )
        self.__tool_scope = ToolScope(policies=DEFAULT_TOOL_POLICIES)
        self.__log_tool_scope_matrix()

        self.__reasoner = reasoner or Reasoner(intent=intent, opener_policy=OpenerSignalPolicy())
        self.__agent_state = agent_state or AgentState(
            intent=intent,
            max_steps=max_steps,
            capabilities=self.__capabilities,
            retries=configuration.intent.retries,
            realignment_budget=self.__realignment.budget,
        )

        self.__signal = signal
        self.__recorder = recorder
        self.__hitl = HITLService(
            phase=phase,
            signal=signal,
            telemetry=telemetry,
            capabilities=self.__capabilities,
        )

        self.__artifact_pipeline = artifact_pipeline

        self.__perception = perception_service or PerceptionService(
            storage=storage,
            perception=perception,
            session_id=workflow_id,
            pipeline=artifact_pipeline,
            hierarchy_signature_builder=HierarchySignatureBuilder(),
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
            tool_scope=self.__tool_scope,
            capabilities=self.__capabilities,
            use_cache=configuration.llm.use_cache,
        )

        self.__capture_store = CaptureStore()
        self.__catalog = CommandCatalogProvider().build()
        self.__action_executor = action_executor or ActionExecutor(
            device=device,
            storage=storage,
            telemetry=telemetry,
            catalog=self.__catalog,
            path_manager=path_manager,
            pipeline=artifact_pipeline,
            capture_store=self.__capture_store,
        )

        self.__comparator = comparator or ScreenComparator()
        self.__hierarchy = hierarchy or HierarchyService(
            storage=storage,
            pipeline=artifact_pipeline,
            configuration=perception_configuration,
        )
        self.__planner = planner or StepPlanner(
            vision_tool=self.__vision,
            tool_scope=self.__tool_scope,
            action_builder=ActionBuilder(),
            escalation_policy=configuration.intent.escalation,
            command_gate=CommandGate(catalog=self.__catalog),
        )

        dialect = DrizzDialectFactory().create()
        execution_evidence = HistoryEvidenceSource(
            distiller=Distiller(),
            path_manager=path_manager,
            assembler=EvidenceAssembler(),
            normalizer=RunTraceNormalizer(classifier=LauncherClassifier()),
        )
        self.__evidence = execution_evidence
        self.__authoring_evidence_builder = AuthoringEvidenceBuilder()

        generation = ScriptGenerationService(
            policy=Policy(),
            dialect=dialect,
            evidence=execution_evidence,
            binder=LaunchBinder(),
            generator=LlmFlowGenerator(
                llm=llm, prompt=FlowPromptBuilder(), use_cache=configuration.llm.use_cache
            ),
        )

        refresher = BaselineRefresher(
            source=execution_evidence,
            path_manager=path_manager,
            baseline=BaselineScriptService(
                policy=Policy(),
                dialect=dialect,
                generator=DeterministicFlowGenerator(),
            ),
        )

        self.__history = history or HistoryService(
            storage=storage,
            refresher=refresher,
            generation=generation,
            workflow_id=workflow_id,
            package_name=package_name,
            path_manager=path_manager,
            pipeline=artifact_pipeline,
        )
        self.__authoring_runner = AuthoringRunner(
            agent=AuthoringAgent(),
            configuration=configuration.authoring,
        )
        self.__authoring = authoring or AuthoringService(
            llm=llm,
            policy=Policy(),
            dialect=dialect,
            use_cache=configuration.llm.use_cache,
            attempts=configuration.authoring.attempts,
        )
        self.__trace = trace or TraceService(path_manager=path_manager)
        self.__resolution = resolution or ReferenceResolutionService(
            ledger=memory,
            catalog=self.__catalog,
            workflow_id=workflow_id,
        )
        self.__artifact_catalog = ArtifactCatalog(path_manager=path_manager)

        self.__ocr = ocr or NoopOcr()
        self.__icons = icons or NoopIconDetector()
        self.__pixel_overlay = pixel_overlay or NoopOverlayDetector()
        self.__ensemble = ensemble or EnsembleLocalizerService(workflow_id=workflow_id)

        self.__screen_observer = screen_observer or ScreenObservationService(
            device=device,
            ocr=self.__ocr,
            icons=self.__icons,
            workflow_id=workflow_id,
            pipeline=artifact_pipeline,
            pixel_overlay=self.__pixel_overlay,
            configuration=perception_configuration,
        )
        self.__target_localizer = target_localizer or TargetLocalizationService(
            catalog=self.__catalog,
            workflow_id=workflow_id,
            ensemble=self.__ensemble,
        )

        self.__abort_detector = abort_detector or AbortDetectorFactory.build(llm=llm)

        self.__journal = journal if journal is not None else NoopRuntimeJournal()

        self.__event_emitter = RuntimeEventEmitter(
            journal=self.__journal,
            workflow_id=workflow_id,
        )

    @staticmethod
    def __resolve_supports_back(*, device: DevicePort) -> bool:
        """
        Return whether the live device adapter can dispatch a system back action.
        """

        runtime = device.configuration
        if runtime is None:
            return True

        return runtime.platform is not DevicePlatform.IOS

    def __log_tool_scope_matrix(self) -> None:
        """
        Dump every (mode-set, hitl) → tool-set expansion at construction time.
        """

        logger.info(
            "Tool scope matrix resolved at boot",
            extra={
                "component": "strategies.graph.context",
                "event": "tool_scope.matrix.resolved",
                "tool_scope.expansions": [
                    expansion.model_dump(mode="json") for expansion in self.__matrix_expansions()
                ],
            },
        )

    def __matrix_expansions(self) -> Sequence[ToolScopeMatrixExpansion]:
        """
        Compute one expansion entry per (mode-set, hitl) combination for the boot log.
        """

        return tuple(
            self.__expansion(modes=modes, hitl=hitl)
            for modes in self.__mode_subsets()
            for hitl in (False, True)
        )

    @staticmethod
    def __mode_subsets() -> Sequence[FrozenSet[TurnMode]]:
        """
        Enumerate every active mode-set combination supported by :class:`TurnMode`.
        """

        modes: Tuple[TurnMode, ...] = tuple(TurnMode.__members__.values())

        return tuple(
            frozenset(combo)
            for size in range(len(modes) + 1)
            for combo in itertools.combinations(modes, size)
        )

    def __expansion(self, *, modes: FrozenSet[TurnMode], hitl: bool) -> ToolScopeMatrixExpansion:
        """
        Resolve one matrix entry for the given mode set and HITL capability.
        """

        capabilities = RuntimeCapabilities(
            hitl=HITLCapability(enabled=hitl),
            device=self.__capabilities.device,
        )
        result = self.__tool_scope.compute(
            context=ToolPolicyContext(capabilities=capabilities, modes=modes),
        )
        return ToolScopeMatrixExpansion(
            hitl=hitl,
            modes=modes,
            tools_allowed=result.names,
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
    def abort_detector(self) -> AbortDetectorPort:
        """
        Composite operator-abort detector wired with LLM primary and heuristic fallback.
        """

        return self.__abort_detector

    @property
    def embedder(self) -> Optional[EmbeddingPort]:
        """
        Embedding port; ``None`` when embedding support is unavailable.
        """

        return self.__embedder

    @property
    def embedding_cache(self) -> Optional[EmbeddingCache]:
        """
        Async cache that warms sub-goal embeddings; ``None`` when disabled.
        """

        return self.__embedding_cache

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
    def phase(self) -> PhaseAnnouncer:
        """
        Returns the PhaseAnnouncer shared with the parent strategy.
        """

        return self.__phase

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
    def capabilities(self) -> RuntimeCapabilities:
        """
        Returns live RuntimeCapabilities derived from the signal port.
        """

        return self.__capabilities

    @property
    def tool_scope(self) -> ToolScope:
        """
        Returns the shared ToolScope.
        """

        return self.__tool_scope

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
    def execution_id(self) -> str:
        """
        Returns the conversation ledger execution identifier.
        """

        return self.__execution_id

    @property
    def tenant(self) -> str:
        """
        Returns the tenant that owns recorded conversation data.
        """

        return self.__tenant

    @property
    def thread(self) -> str:
        """
        Returns the conversation thread identifier for recording.
        """

        return self.__thread

    @property
    def workspace(self) -> Optional[str]:
        """
        Returns the optional workspace boundary for recording.
        """

        return self.__workspace

    @property
    def requester(self) -> str:
        """
        Returns the actor that requested the workflow.
        """

        return self.__requester

    @property
    def responder(self) -> str:
        """
        Returns the actor that responds for the workflow.
        """

        return self.__responder

    @property
    def recorder(self) -> Optional[ConversationRecorder]:
        """
        Returns the optional conversation recorder.
        """

        return self.__recorder

    @property
    def package_name(self) -> str:
        """
        Returns the set package name.
        """

        return self.__package_name

    @property
    def artifact_catalog(self) -> ArtifactCatalog:
        """
        Returns the artifact catalog bound to the shared path manager.
        """

        return self.__artifact_catalog

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
    def catalog(self) -> CommandCatalog:
        """
        Returns the shared command capability catalog.
        """

        return self.__catalog

    @property
    def capture_store(self) -> CaptureStore:
        """
        Returns the run-owned capture store shared with the action executor.
        """

        return self.__capture_store

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
    def authoring_runner(self) -> AuthoringRunner:
        """
        Returns the script authoring runner.
        """

        return self.__authoring_runner

    @property
    def authoring(self) -> AuthoringPort:
        """
        Returns the authoring port used by the authoring runner to produce scripts.
        """

        return self.__authoring

    @property
    def evidence(self) -> EvidenceSource:
        """
        Returns the execution evidence source used by script authoring.
        """

        return self.__evidence

    @property
    def authoring_evidence_builder(self) -> AuthoringEvidenceBuilder:
        """
        Returns the builder that derives authoring task evidence from execution evidence.
        """

        return self.__authoring_evidence_builder

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

        try:
            await self.__phase.shutdown()
        except Exception as exception:
            logger.warning(f"[graph-context] phase announcer shutdown failed: {exception}")

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
