from __future__ import annotations

import time
import uuid
from logging import getLogger
from typing import Any, Dict, Optional

from fathom.adapters.summarization.llm import LLMSummarizer
from fathom.base.paths import SharedPathManager
from fathom.core.context.manager import ContextManager
from fathom.core.execution.engine import ExecutionEngine
from fathom.interfaces.device import DevicePort
from fathom.interfaces.knowledge import KnowledgePort
from fathom.interfaces.llm import LLMPort
from fathom.interfaces.memory import MemoryPort
from fathom.interfaces.signal import SignalPort
from fathom.interfaces.storage import StoragePort
from fathom.interfaces.telemetry import TelemetryPort
from fathom.schemas.configuration import FathomConfiguration
from fathom.schemas.exploration import ExplorationGraph
from fathom.schemas.orchestration import RealignmentPolicy
from fathom.schemas.results import ExplorationResult, IntentResult
from fathom.strategies.exploration import ExplorationStrategy
from fathom.strategies.intent import IntentStrategy

logger = getLogger(__name__)


class FathomRunner:
    """
    Executes Fathom workflows with configured ports.

    This is the main execution orchestrator that wires together all ports
    and coordinates the execution of automation workflows using hexagonal architecture.

    The runner:
    - Wires ExecutionEngine and ContextManager
    - Manages execution lifecycle
    - Delegates to strategy implementations
    - Returns results compatible with CLI expectations
    """

    def __init__(
        self,
        *,
        llm: LLMPort,
        device: DevicePort,
        memory: MemoryPort,
        signal: SignalPort,
        storage: StoragePort,
        knowledge: KnowledgePort,
        telemetry: TelemetryPort,
        path_manager: SharedPathManager,
        config: Optional[FathomConfiguration] = None,
        realignment: Optional[RealignmentPolicy] = None,
    ) -> None:
        """
        Initialize runner with all configured ports.
        """

        self.__llm = llm
        self.__device = device

        self.__memory = memory
        self.__knowledge = knowledge

        self.__signal = signal
        self.__storage = storage
        self.__telemetry = telemetry
        self.__path_manager = path_manager
        self.__config = config or FathomConfiguration()
        self.__realignment = realignment or RealignmentPolicy()

        # Wire core components
        self.__engine = ExecutionEngine(
            llm=llm,
            device=device,
            memory=memory,
            signal=signal,
            storage=storage,
            telemetry=telemetry,
            path_manager=path_manager,
        )
        self.__context_manager: Optional[ContextManager] = None

        # Track current workflow for cancellation
        self.__current_strategy: Optional[object] = None

    @property
    def engine(self) -> ExecutionEngine:
        """
        Get the execution engine.
        """

        return self.__engine

    @property
    def context(self) -> Optional[ContextManager]:
        """
        Get the context manager.
        """

        return self.__context_manager

    async def run_intent(
        self,
        intent: str,
        max_steps: int = 50,
        use_xml: bool = False,
        request_id: Optional[str] = None,
        realignment: Optional[RealignmentPolicy] = None,
    ) -> IntentResult:
        """
        Execute intent-based workflow.
        """

        start_time = time.time()
        workflow_id = request_id or uuid.uuid4().hex[:8]

        try:
            package_name = await self.__device.get_current_package()
        except Exception as exception:
            package_name = "unknown_app"
            await self.__telemetry.warning(
                "Failed to get package name, using fallback", error=str(exception)
            )

        if self.__device.configuration:
            device_serial = self.__device.configuration.serial_number
        else:
            device_serial = None

        await self.__telemetry.info(
            "Starting intent workflow",
            intent=intent,
            max_steps=max_steps,
            workflow_id=workflow_id,
            package_name=package_name,
            device_serial=device_serial,
        )

        # Initialize context
        self.__context_manager = ContextManager(memory=self.__memory, workflow_id=workflow_id)
        self.__context_manager.set_roadmap(intent=intent)

        # Create and execute strategy
        strategy = IntentStrategy(
            intent=intent,
            llm=self.__llm,
            device=self.__device,
            memory=self.__memory,
            signal=self.__signal,
            storage=self.__storage,
            workflow_id=workflow_id,
            package_name=package_name,
            telemetry=self.__telemetry,
            configuration=self.__config,
            path_manager=self.__path_manager,
            summarizer=LLMSummarizer(llm=self.__llm),
            realignment=realignment or self.__realignment,
            max_steps=max_steps or self.__config.intent.max_steps,
            use_xml=use_xml if use_xml is not None else self.__config.intent.use_xml_grounding,
        )
        self.__current_strategy = strategy

        try:
            # Execute strategy
            execution_result = await strategy.execute()

            # Get progress info
            progress = strategy.get_progress()

            # Collect metrics from strategy - use to_report_dict() for proper format
            strategy_metrics = strategy.get_metrics()
            metrics = strategy_metrics.to_report_dict() if strategy_metrics else {}

            # Get memory summary
            memory_summary = await self.__get_memory_summary()

            # Build IntentResult
            duration = time.time() - start_time
            completion_reason = "Completed successfully" if execution_result.success else "Failed"

            result = IntentResult(
                intent=intent,
                metrics=metrics,
                duration=duration,
                workflow_id=workflow_id,
                error=execution_result.error,
                memory_summary=memory_summary,
                success=execution_result.success,
                completion_reason=completion_reason,
                steps_taken=progress.get("step_count", 0),
                steps_executed=progress.get("step_count", 0),
                status="completed" if execution_result.success else "failed",
            )

            await self.__telemetry.info(
                "Intent workflow completed",
                duration=duration,
                success=result.success,
                steps_taken=result.steps_taken,
            )

            return result

        finally:
            self.__current_strategy = None

    async def run_exploration(
        self,
        max_steps: int = 100,
        request_id: Optional[str] = None,
    ) -> ExplorationResult:
        """
        Execute exploration workflow.
        """

        start_time = time.time()
        workflow_id = request_id or uuid.uuid4().hex[:8]

        # Fetch package name from device at start
        try:
            package_name = await self.__device.get_current_package()
        except Exception as exception:
            package_name = "unknown_app"
            await self.__telemetry.warning(
                "Failed to get package name, using fallback", error=str(exception)
            )

        if self.__device.configuration:
            device_serial = self.__device.configuration.serial_number
        else:
            device_serial = None

        await self.__telemetry.info(
            "Starting exploration workflow",
            max_steps=max_steps,
            workflow_id=workflow_id,
            package_name=package_name,
            device_serial=device_serial,
        )

        # Initialize context
        self.__context_manager = ContextManager(memory=self.__memory, workflow_id=workflow_id)
        self.__context_manager.set_roadmap(intent="Explore application structure")

        strategy = ExplorationStrategy(
            llm=self.__llm,
            device=self.__device,
            memory=self.__memory,
            signal=self.__signal,
            storage=self.__storage,
            workflow_id=workflow_id,
            package_name=package_name,
            telemetry=self.__telemetry,
            configuration=self.__config,
            path_manager=self.__path_manager,
            seed=self.__config.exploration.random_seed,
            timeout=float(self.__config.exploration.timeout),
            max_steps=max_steps or self.__config.exploration.max_steps,
        )

        self.__current_strategy = strategy

        try:
            # Execute strategy
            execution_result = await strategy.execute()

            # Get progress info
            progress = strategy.get_progress()
            stats = progress.get("stats", {})

            # Extract discovered activities from graph
            graph = strategy.graph
            discovered_activities = list({node.activity for node in graph.nodes.values()})

            # Calculate coverage (percentage of screens explored vs total discovered)
            unique_screens = stats.get("unique_screens", 0)
            unexplored = stats.get("unexplored", 0)
            coverage_percentage = (
                ((unique_screens - unexplored) / unique_screens * 100.0)
                if unique_screens > 0
                else 0.0
            )

            # Export graph structure
            screen_graph = await self.__export_graph(graph=graph)

            # Build ExplorationResult
            duration = time.time() - start_time

            result = ExplorationResult(
                duration=duration,
                workflow_id=workflow_id,
                screen_graph=screen_graph,
                error=execution_result.error,
                unique_screens=unique_screens,
                success=execution_result.success,
                steps_executed=progress.get("steps", 0),
                coverage_percentage=coverage_percentage,
                completion_reason="Exploration completed",
                discovered_activities=discovered_activities,
                total_actions=stats.get("total_actions", 0),
                total_transitions=stats.get("total_transitions", 0),
                status="completed" if execution_result.success else "failed",
            )

            await self.__telemetry.info(
                "Exploration workflow completed",
                duration=duration,
                total_actions=result.total_actions,
                unique_screens=result.unique_screens,
            )

            return result

        finally:
            self.__current_strategy = None

    def cancel(self) -> None:
        """
        Cancel the currently running workflow.
        """

        if self.__current_strategy:
            logger.warning("Workflow cancellation requested")

            # Call cancel method on strategy if it has one
            if hasattr(self.__current_strategy, "cancel"):
                self.__current_strategy.cancel()
            else:
                logger.warning("Strategy does not support cancellation")

    async def cleanup(self) -> None:
        """
        Cleanup resources.
        """

        try:
            await self.__llm.cleanup()
        except Exception as exception:
            await self.__telemetry.warning(f"LLM cleanup failed: {exception}")

        await self.__telemetry.info("Runner cleanup completed")

    async def __get_memory_summary(self) -> Dict[str, Any]:
        """
        Get memory summary from memory port.
        """

        try:
            # Get all knowledge from memory provider
            knowledge = await self.__memory.get_all_knowledge()

            # Extract screens information
            screens = knowledge.get("screens", [])

            # Format for CLI display
            screens_formatted = []
            for screen in screens[:10]:  # Last 10 screens
                screens_formatted.append(
                    {
                        "hash": screen.get("hash", "")[:12],
                        "activity": screen.get("activity", "unknown"),
                        "description": screen.get("description", ""),
                    }
                )

            # Count experiences
            experience_count = knowledge.get("experience_count", 0)

            return {
                "screens": screens_formatted,
                "total_screens": len(screens),
                "experience_count": experience_count,
            }
        except Exception as exception:
            await self.__telemetry.warning(f"Failed to get memory summary: {exception}")
            return {
                "screens": [],
                "total_screens": 0,
                "experience_count": 0,
            }

    async def __export_graph(self, graph: ExplorationGraph) -> Dict[str, Any]:
        """
        Export exploration graph to dictionary.
        """

        try:
            nodes_dict = {}

            for fingerprint, node in graph.nodes.items():
                nodes_dict[fingerprint] = {
                    "visits": node.visits,
                    "activity": node.activity,
                    "actions": list(node.actions),
                    "transitions": node.transitions,
                }

            edges_list = [
                {"origin": origin, "action": action, "destination": dest}
                for origin, action, dest in graph.edges
            ]

            return {
                "nodes": nodes_dict,
                "edges": edges_list,
                "stats": graph.get_stats(),
            }
        except Exception as exception:
            await self.__telemetry.warning(f"Failed to export graph: {exception}")
            return {}
