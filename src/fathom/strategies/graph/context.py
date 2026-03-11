from __future__ import annotations

import asyncio
from typing import Optional

from fathom.base.paths import SharedPathManager
from fathom.core.agent.planner import StepPlanner
from fathom.core.agent.reasoner import Reasoner
from fathom.core.agent.state import AgentState
from fathom.core.context.manager import ContextManager
from fathom.core.services.action import ActionExecutor
from fathom.core.services.exporter import ScriptExporter
from fathom.core.services.hierarchy import HierarchyService
from fathom.core.services.history import HistoryService
from fathom.core.services.hitl import HITLService
from fathom.core.services.perception import PerceptionService
from fathom.core.services.resolution import ReferenceResolutionService
from fathom.core.services.trace import TraceService
from fathom.core.services.vision import VisionService
from fathom.interfaces.device import DevicePort
from fathom.interfaces.knowledge import KnowledgePort
from fathom.interfaces.llm import LLMPort
from fathom.interfaces.memory import MemoryPort
from fathom.interfaces.perception import PerceptionPort
from fathom.interfaces.signal import SignalPort
from fathom.interfaces.storage import StoragePort
from fathom.interfaces.summarization import SummarizationPort
from fathom.interfaces.telemetry import TelemetryPort
from fathom.schemas.configuration import FathomConfiguration
from fathom.schemas.exploration import ExplorationGraph
from fathom.schemas.metrics import ExecutionMetrics
from fathom.schemas.run import RealignmentPolicy


class GraphContext:
    """
    Context container for graph nodes.

    Holds ports, services, and mutable agent state.
    """

    def __init__(
        self,
        intent: str,
        llm: LLMPort,
        device: DevicePort,
        perception: PerceptionPort,
        memory: MemoryPort,
        signal: SignalPort,
        storage: StoragePort,
        telemetry: TelemetryPort,
        path_manager: SharedPathManager,
        configuration: FathomConfiguration,
        *,
        use_xml: bool,
        max_steps: int,
        workflow_id: str,
        package_name: str,
        reasoner: Optional[Reasoner] = None,
        trace: Optional[TraceService] = None,
        planner: Optional[StepPlanner] = None,
        vision: Optional[VisionService] = None,
        agent_state: Optional[AgentState] = None,
        history: Optional[HistoryService] = None,
        knowledge: Optional[KnowledgePort] = None,
        metrics: Optional[ExecutionMetrics] = None,
        cancel_event: Optional[asyncio.Event] = None,
        hierarchy: Optional[HierarchyService] = None,
        perception_service: Optional[PerceptionService] = None,
        summarizer: Optional[SummarizationPort] = None,
        realignment: Optional[RealignmentPolicy] = None,
        context_manager: Optional[ContextManager] = None,
        action_executor: Optional[ActionExecutor] = None,
        exploration_graph: Optional[ExplorationGraph] = None,
        resolution: Optional[ReferenceResolutionService] = None,
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

        self.__perception = perception_service or PerceptionService(
            perception=perception,
            storage=storage,
            session_id=workflow_id,
        )

        # GCC Context Manager with optional summarizer
        self.__context_manager = context_manager or ContextManager(
            memory=memory, workflow_id=workflow_id, summarizer=summarizer
        )

        self.__vision = vision or VisionService(
            llm=llm,
            memory=memory,
            storage=storage,
            telemetry=telemetry,
            session_id=workflow_id,
            package_name=package_name,
            use_cache=configuration.llm.use_cache,
        )

        self.__action_executor = action_executor or ActionExecutor(
            device=device,
            storage=storage,
            telemetry=telemetry,
            path_manager=path_manager,
        )

        self.__hierarchy = hierarchy or HierarchyService(storage=storage)
        self.__planner = planner or StepPlanner(vision_tool=self.__vision)

        self.__history = history or HistoryService(
            storage=storage,
            workflow_id=workflow_id,
            package_name=package_name,
            path_manager=path_manager,
            exporter=ScriptExporter(llm=llm, use_cache=configuration.llm.use_cache),
        )
        self.__trace = trace or TraceService(path_manager=path_manager)
        self.__resolution = resolution or ReferenceResolutionService(ledger=memory)

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

        Args:
            state: New AgentState instance to use
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
