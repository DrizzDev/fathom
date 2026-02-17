"""Fathom execution runner."""

from __future__ import annotations

import time
import uuid
from typing import TYPE_CHECKING, Any, Dict, Optional

from fathom.core.context.manager import ContextManager
from fathom.core.execution.engine import ExecutionEngine
from fathom.schemas.configuration import FathomConfig
from fathom.schemas.orchestration import RealignmentPolicy
from fathom.schemas.results import ExplorationResult, IntentResult

if TYPE_CHECKING:
    from fathom.base.paths import SharedPathManager
    from fathom.interfaces.device import DevicePort
    from fathom.interfaces.knowledge import KnowledgePort
    from fathom.interfaces.llm import LLMPort
    from fathom.interfaces.memory import MemoryPort
    from fathom.interfaces.signal import SignalPort
    from fathom.interfaces.storage import StoragePort
    from fathom.interfaces.telemetry import TelemetryPort
    from fathom.schemas.exploration import ExplorationGraph


class FathomRunner:
    """
    Executes Fathom workflows with configured ports.

    This is the main execution orchestrator that wires together all ports
    and coordinates the execution of automation workflows using the new
    hexagonal architecture.

    The runner:
    - Wires ExecutionEngine and ContextManager
    - Manages execution lifecycle
    - Delegates to strategy implementations
    - Returns results compatible with CLI expectations
    """

    def __init__(
        self,
        *,
        device: DevicePort,
        llm: LLMPort,
        memory: MemoryPort,
        knowledge: KnowledgePort,
        signal: SignalPort,
        storage: StoragePort,
        telemetry: TelemetryPort,
        path_manager: SharedPathManager,
        config: Optional[FathomConfig] = None,
    ) -> None:
        """
        Initialize runner with all configured ports.

        Args:
            device: Device port for mobile device interactions
            llm: LLM port for reasoning and analysis
            memory: Memory port for session state and cross-run memory
            knowledge: Knowledge port for application knowledge graph
            signal: Signal port for human-in-the-loop control
            storage: Storage port for artifact persistence
            telemetry: Telemetry port for logging and observability
            path_manager: Shared path manager for trace storage
            config: Optional configuration (uses defaults if not provided)
        """
        self.__device = device
        self.__llm = llm
        self.__memory = memory
        self.__knowledge = knowledge
        self.__signal = signal
        self.__storage = storage
        self.__telemetry = telemetry
        self.__path_manager = path_manager
        self.__config = config or FathomConfig()

        # Wire core components
        self.__engine = ExecutionEngine(
            device=device,
            llm=llm,
            memory=memory,
            signal=signal,
            storage=storage,
            telemetry=telemetry,
            path_manager=path_manager,
        )
        self.__context_manager: Optional[ContextManager] = None

        # Track current workflow for cancellation
        self.__current_strategy: Optional[object] = None

    async def run_intent(
        self,
        intent: str,
        max_steps: int = 20,
        use_xml: bool = False,
        request_id: Optional[str] = None,
        device_serial: Optional[str] = None,
        prompt_version: Optional[str] = None,
        realignment: Optional[RealignmentPolicy] = None,
    ) -> IntentResult:
        """
        Execute intent-based workflow.

        Args:
            intent: User intent to accomplish
            max_steps: Maximum execution steps
            use_xml: Whether to use XML hierarchy
            request_id: Optional workflow ID
            device_serial: Device serial (unused, kept for compatibility)
            prompt_version: Prompt version (unused, kept for compatibility)
            realignment: Configuration for HITL re-planning behavior

        Returns:
            IntentResult with execution outcome and metrics
        """
        workflow_id = request_id or uuid.uuid4().hex[:8]
        start_time = time.time()

        # Fetch package name from device at start
        try:
            package_name = await self.__device.get_current_package()
        except Exception:
            package_name = "unknown_app"

        self.__telemetry.info(
            "Starting intent workflow",
            intent=intent,
            max_steps=max_steps,
            workflow_id=workflow_id,
            package_name=package_name,
        )

        # Initialize context
        self.__context_manager = ContextManager(memory=self.__memory, workflow_id=workflow_id)
        self.__context_manager.set_roadmap(intent=intent)

        # Create and execute strategy
        from fathom.adapters.summarization.llm import LLMSummarizer
        from fathom.strategies.intent import IntentStrategy

        summarizer = LLMSummarizer(llm=self.__llm)

        strategy = IntentStrategy(
            intent=intent,
            device=self.__device,
            llm=self.__llm,
            memory=self.__memory,
            storage=self.__storage,
            telemetry=self.__telemetry,
            signal=self.__signal,
            summarizer=summarizer,
            path_manager=self.__path_manager,
            max_steps=max_steps or self.__config.intent_strategy.max_steps,
            use_xml=use_xml if use_xml is not None else self.__config.intent_strategy.use_xml,
            workflow_id=workflow_id,
            package_name=package_name,
            realignment=realignment,
        )
        self.__current_strategy = strategy

        try:
            # Execute strategy
            execution_result = await strategy.execute(max_steps=max_steps)

            # Get progress info
            progress = strategy.get_progress()

            # Collect metrics from strategy - use to_report_dict() for proper format
            strategy_metrics = strategy.get_metrics()
            metrics = strategy_metrics.to_report_dict() if strategy_metrics else {}

            # Get memory summary
            memory_summary = await self.__get_memory_summary()

            # Build IntentResult
            duration = time.time() - start_time

            result = IntentResult(
                success=execution_result.success,
                completion_reason=execution_result.error
                or ("Completed successfully" if execution_result.success else "Failed"),
                workflow_id=workflow_id,
                status="completed" if execution_result.success else "failed",
                duration=duration,
                intent=intent,
                steps_taken=progress.get("step_count", 0),
                steps_executed=progress.get("step_count", 0),
                metrics=metrics,
                memory_summary=memory_summary,
                error=execution_result.error,
            )

            self.__telemetry.info(
                "Intent workflow completed",
                success=result.success,
                steps_taken=result.steps_taken,
                duration=duration,
            )

            return result

        finally:
            self.__current_strategy = None

    async def run_exploration(
        self,
        max_steps: int = 50,
        request_id: Optional[str] = None,
        device_serial: Optional[str] = None,
    ) -> ExplorationResult:
        """
        Execute exploration workflow.

        Args:
            max_steps: Maximum exploration steps
            request_id: Optional workflow ID
            device_serial: Device serial (unused, kept for compatibility)

        Returns:
            ExplorationResult with discovery metrics
        """
        workflow_id = request_id or uuid.uuid4().hex[:8]
        start_time = time.time()

        # Fetch package name from device at start
        try:
            package_name = await self.__device.get_current_package()
        except Exception:
            package_name = "unknown_app"

        self.__telemetry.info(
            "Starting exploration workflow",
            max_steps=max_steps,
            workflow_id=workflow_id,
            package_name=package_name,
        )

        # Initialize context
        self.__context_manager = ContextManager(memory=self.__memory, workflow_id=workflow_id)
        self.__context_manager.set_roadmap(intent="Explore application structure")

        # Create and execute strategy
        from fathom.strategies.exploration import ExplorationStrategy

        strategy = ExplorationStrategy(
            device=self.__device,
            llm=self.__llm,
            memory=self.__memory,
            storage=self.__storage,
            telemetry=self.__telemetry,
            signal=self.__signal,
            path_manager=self.__path_manager,
            max_steps=max_steps or self.__config.exploration_strategy.max_steps,
            timeout=self.__config.exploration_strategy.timeout,
            seed=self.__config.exploration_strategy.seed,
            package_name=package_name,
            workflow_id=workflow_id,
        )
        self.__current_strategy = strategy

        try:
            # Execute strategy
            execution_result = await strategy.execute(
                max_steps=max_steps or self.__config.exploration_strategy.max_steps
            )

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
            screen_graph = self.__export_graph(graph=graph)

            # Build ExplorationResult
            duration = time.time() - start_time

            result = ExplorationResult(
                success=execution_result.success,
                completion_reason="Exploration completed",
                workflow_id=workflow_id,
                status="completed" if execution_result.success else "failed",
                duration=duration,
                steps_executed=progress.get("steps", 0),
                unique_screens=unique_screens,
                total_actions=stats.get("total_actions", 0),
                total_transitions=stats.get("total_transitions", 0),
                coverage_percentage=coverage_percentage,
                discovered_activities=discovered_activities,
                screen_graph=screen_graph,
                error=execution_result.error,
            )

            self.__telemetry.info(
                "Exploration workflow completed",
                unique_screens=result.unique_screens,
                total_actions=result.total_actions,
                duration=duration,
            )

            return result

        finally:
            self.__current_strategy = None

    def cancel(self) -> None:
        """Cancel the currently running workflow."""
        if self.__current_strategy:
            self.__telemetry.warning("Workflow cancellation requested")
            # Call cancel method on strategy if it has one
            if hasattr(self.__current_strategy, "cancel"):
                self.__current_strategy.cancel()
            else:
                self.__telemetry.warning("Strategy does not support cancellation")

    async def cleanup(self) -> None:
        """Cleanup resources."""
        # Cleanup LLM resources
        await self.__llm.cleanup()

        self.__telemetry.info("Runner cleanup completed")

    async def __get_memory_summary(self) -> Dict[str, Any]:
        """Get memory summary from memory port."""
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
                "experience_count": experience_count,
                "total_screens": len(screens),
            }
        except Exception as exception:
            self.__telemetry.warning(f"Failed to get memory summary: {exception}")
            return {
                "screens": [],
                "experience_count": 0,
                "total_screens": 0,
            }

    def __export_graph(self, graph: ExplorationGraph) -> Dict[str, Any]:
        """Export exploration graph to dictionary."""
        try:
            nodes_dict = {}
            for fingerprint, node in graph.nodes.items():
                nodes_dict[fingerprint] = {
                    "activity": node.activity,
                    "visits": node.visits,
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
            self.__telemetry.warning(f"Failed to export graph: {exception}")
            return {}

    @property
    def engine(self) -> ExecutionEngine:
        """Get the execution engine."""
        return self.__engine

    @property
    def context(self) -> Optional[ContextManager]:
        """Get the context manager."""
        return self.__context_manager
