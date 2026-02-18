from __future__ import annotations

import asyncio
from typing import Optional

from fathom.base.paths import SharedPathManager
from fathom.core.agent.planner import StepPlanner
from fathom.core.agent.reasoner import Reasoner
from fathom.core.agent.state import AgentState
from fathom.core.context.manager import ContextManager
from fathom.core.services.audit import AuditService
from fathom.core.services.hierarchy import HierarchyService
from fathom.core.services.history import HistoryService
from fathom.core.services.resolution import ReferenceResolutionService
from fathom.core.services.trace import TraceService
from fathom.core.services.ux import UXService
from fathom.core.services.vision import VisionService
from fathom.interfaces.device import DevicePort
from fathom.interfaces.knowledge import KnowledgePort
from fathom.interfaces.llm import LLMPort
from fathom.interfaces.memory import MemoryPort
from fathom.interfaces.signal import SignalPort
from fathom.interfaces.storage import StoragePort
from fathom.interfaces.summarization import SummarizationPort
from fathom.interfaces.telemetry import TelemetryPort
from fathom.schemas.exploration import ExplorationGraph
from fathom.schemas.metrics import ExecutionMetrics
from fathom.schemas.orchestration import RealignmentPolicy


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
        memory: MemoryPort,
        signal: SignalPort,
        storage: StoragePort,
        telemetry: TelemetryPort,
        path_manager: SharedPathManager,
        *,
        max_steps: int = 20,
        use_xml: bool = False,
        workflow_id: str = "default",
        package_name: str = "unknown_app",
        auditor: Optional[AuditService] = None,
        knowledge: Optional[KnowledgePort] = None,
        cancel_event: Optional[asyncio.Event] = None,
        summarizer: Optional[SummarizationPort] = None,
        realignment: Optional[RealignmentPolicy] = None,
        exploration_graph: Optional[ExplorationGraph] = None,
    ) -> None:
        self.__intent = intent
        self.__device = device

        self.__llm = llm
        self.__memory = memory
        self.__storage = storage
        self.__knowledge = knowledge

        self.__signal = signal
        self.__telemetry = telemetry
        self.__path_manager = path_manager

        self.__exploration_graph = exploration_graph or ExplorationGraph()

        self.__use_xml = use_xml
        self.__max_steps = max_steps
        self.__workflow_id = workflow_id
        self.__package_name = package_name

        self.__cancel_event = cancel_event or asyncio.Event()
        self.__realignment = realignment or RealignmentPolicy()

        self.__ux = UXService()
        self.__metrics = ExecutionMetrics()
        self.__auditor = auditor or AuditService()

        self.__reasoner = Reasoner(intent=intent)
        self.__agent_state = AgentState(intent=intent, max_steps=max_steps)

        # GCC Context Manager with optional summarizer
        self.__context_manager = ContextManager(
            memory=memory, workflow_id=workflow_id, summarizer=summarizer
        )

        self.__vision = VisionService(
            llm=llm,
            memory=memory,
            storage=storage,
            auditor=self.__auditor,
            session_id=workflow_id,
            package_name=package_name,
        )

        self.__hierarchy = HierarchyService(device=device)
        self.__planner = StepPlanner(vision_tool=self.__vision)

        self.__history = HistoryService(
            workflow_id=workflow_id,
            package_name=package_name,
            path_manager=path_manager,
        )
        self.__trace = TraceService(path_manager=path_manager)
        self.__resolution = ReferenceResolutionService(ledger=memory)

    @property
    def intent(self) -> str:
        """
        Returns the original intent
        """

        return self.__intent

    @property
    def device(self) -> DevicePort:
        """
        Returns the `DevicePort` instance
        """

        return self.__device

    @property
    def llm(self) -> LLMPort:
        """
        Returns the `LLMPort` instance
        """

        return self.__llm

    @property
    def memory(self) -> MemoryPort:
        """
        Returns the `MemoryPort` instance
        """

        return self.__memory

    @property
    def knowledge(self) -> Optional[KnowledgePort]:
        """
        Returns the `KnowledgePort` instance
        """

        return self.__knowledge

    @property
    def storage(self) -> StoragePort:
        """
        Returns the `StoragePort` instance
        """

        return self.__storage

    @property
    def telemetry(self) -> TelemetryPort:
        """
        Returns the `TelemetryPort` instance
        """

        return self.__telemetry

    @property
    def signal(self) -> SignalPort:
        """
        Returns the `SignalPort` instance
        """

        return self.__signal

    @property
    def path_manager(self) -> SharedPathManager:
        """
        Returns the `SharedPathManager` instance
        """

        return self.__path_manager

    @property
    def max_steps(self) -> int:
        """
        Returns the maximum number of steps allowed
        """

        return self.__max_steps

    @property
    def use_xml(self) -> bool:
        """
        Boolean value indicating whether to use XML in grounding
        """

        return self.__use_xml

    @property
    def workflow_id(self) -> str:
        """
        Returns the workflow ID
        """

        return self.__workflow_id

    @property
    def package_name(self) -> str:
        """
        Returns the set package name
        """

        return self.__package_name

    @property
    def realignment(self) -> RealignmentPolicy:
        """
        Returns the `RealignmentPolicy` instance
        """

        return self.__realignment

    @property
    def is_cancelled(self) -> bool:
        """
        Boolean value indicating whether the context is cancelled
        """

        return self.__cancel_event.is_set()

    def cancel(self) -> None:
        """
        Signal cancellation.
        """

        self.__cancel_event.set()

    @property
    def exploration_graph(self) -> ExplorationGraph:
        """
        Returns the `ExplorationGraph` instance
        """

        return self.__exploration_graph

    @property
    def agent_state(self) -> AgentState:
        """
        Returns the `AgentState` instance
        """

        return self.__agent_state

    @property
    def context_manager(self) -> ContextManager:
        """
        Returns the `ContextManager` instance
        """

        return self.__context_manager

    @property
    def reasoner(self) -> Reasoner:
        """
        Returns the `Reasoner` instance
        """

        return self.__reasoner

    @property
    def metrics(self) -> ExecutionMetrics:
        """
        Returns the `ExecutionMetrics` instance
        """

        return self.__metrics

    @property
    def vision(self) -> VisionService:
        """
        Returns the `VisionService` instance
        """

        return self.__vision

    @property
    def planner(self) -> StepPlanner:
        """
        Returns the `StepPlanner` instance
        """

        return self.__planner

    @property
    def hierarchy(self) -> HierarchyService:
        """
        Returns the `HierarchyService` instance
        """

        return self.__hierarchy

    @property
    def history(self) -> HistoryService:
        """
        Returns the `HistoryService` instance
        """

        return self.__history

    @property
    def trace(self) -> TraceService:
        """
        Returns the `TraceService` instance
        """

        return self.__trace

    @property
    def auditor(self) -> AuditService:
        """
        Returns the `AuditService` instance
        """

        return self.__auditor

    @property
    def ux(self) -> UXService:
        """
        Returns the `UXService` instance
        """
        return self.__ux

    @property
    def resolution(self) -> ReferenceResolutionService:
        """
        Returns the `ReferenceResolutionService` instance
        """

        return self.__resolution
