"""
Shared context for graph execution.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Optional

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
from fathom.interfaces.telemetry import TelemetryPort
from fathom.schemas.exploration import ExplorationGraph
from fathom.schemas.metrics import ExecutionMetrics

if TYPE_CHECKING:
    from fathom.base.paths import SharedPathManager


class GraphContext:
    """
    Context container for graph nodes.
    Holds ports, services, and mutable agent state.
    """

    def __init__(
        self,
        intent: str,
        device: DevicePort,
        llm: LLMPort,
        memory: MemoryPort,
        storage: StoragePort,
        telemetry: TelemetryPort,
        signal: SignalPort,
        path_manager: SharedPathManager,
        *,
        knowledge: Optional[KnowledgePort] = None,
        max_steps: int = 20,
        use_xml: bool = False,
        workflow_id: str = "default",
        package_name: str = "unknown_app",
        cancel_event: Optional[asyncio.Event] = None,
        exploration_graph: Optional[ExplorationGraph] = None,
        auditor: Optional[AuditService] = None,
    ) -> None:
        self.__intent = intent
        self.__device = device
        self.__llm = llm
        self.__memory = memory
        self.__storage = storage
        self.__telemetry = telemetry
        self.__signal = signal
        self.__path_manager = path_manager
        self.__knowledge = knowledge
        self.__exploration_graph = exploration_graph or ExplorationGraph()

        self.__max_steps = max_steps
        self.__use_xml = use_xml
        self.__workflow_id = workflow_id
        self.__package_name = package_name
        self.__cancel_event = cancel_event or asyncio.Event()

        # --- Domain Services & Agent ---
        self.__agent_state = AgentState(intent=intent, max_steps=max_steps)
        self.__reasoner = Reasoner(intent=intent)
        self.__metrics = ExecutionMetrics()
        self.__context_manager = ContextManager(memory=memory, workflow_id=workflow_id)

        # --- Application Services ---
        self.__auditor = auditor or AuditService()

        self.__vision = VisionService(
            llm=llm,
            memory=memory,
            storage=storage,
            session_id=workflow_id,
            package_name=package_name,
            auditor=self.__auditor,
        )

        self.__planner = StepPlanner(vision_tool=self.__vision)
        self.__hierarchy = HierarchyService(device=device)
        self.__history = HistoryService(
            workflow_id=workflow_id,
            package_name=package_name,
            path_manager=path_manager,
        )
        self.__trace = TraceService(path_manager=path_manager)
        self.__ux = UXService()
        self.__resolution = ReferenceResolutionService(ledger=memory)

    @property
    def intent(self) -> str:
        return self.__intent

    @property
    def device(self) -> DevicePort:
        return self.__device

    @property
    def llm(self) -> LLMPort:
        return self.__llm

    @property
    def memory(self) -> MemoryPort:
        return self.__memory

    @property
    def knowledge(self) -> Optional[KnowledgePort]:
        return self.__knowledge

    @property
    def storage(self) -> StoragePort:
        return self.__storage

    @property
    def telemetry(self) -> TelemetryPort:
        return self.__telemetry

    @property
    def signal(self) -> SignalPort:
        return self.__signal

    @property
    def path_manager(self) -> SharedPathManager:
        return self.__path_manager

    @property
    def max_steps(self) -> int:
        return self.__max_steps

    @property
    def use_xml(self) -> bool:
        return self.__use_xml

    @property
    def workflow_id(self) -> str:
        return self.__workflow_id

    @property
    def package_name(self) -> str:
        return self.__package_name

    @property
    def is_cancelled(self) -> bool:
        return self.__cancel_event.is_set()

    def cancel(self) -> None:
        """Signal cancellation."""
        self.__cancel_event.set()

    @property
    def exploration_graph(self) -> ExplorationGraph:
        return self.__exploration_graph

    # --- Service Accessors ---

    @property
    def agent_state(self) -> AgentState:
        return self.__agent_state

    @property
    def context_manager(self) -> ContextManager:
        return self.__context_manager

    @property
    def reasoner(self) -> Reasoner:
        return self.__reasoner

    @property
    def metrics(self) -> ExecutionMetrics:
        return self.__metrics

    @property
    def vision(self) -> VisionService:
        return self.__vision

    @property
    def planner(self) -> StepPlanner:
        return self.__planner

    @property
    def hierarchy(self) -> HierarchyService:
        return self.__hierarchy

    @property
    def history(self) -> HistoryService:
        return self.__history

    @property
    def trace(self) -> TraceService:
        return self.__trace

    @property
    def auditor(self) -> AuditService:
        return self.__auditor

    @property
    def ux(self) -> UXService:
        return self.__ux

    @property
    def resolution(self) -> ReferenceResolutionService:
        return self.__resolution
