"""Fathom execution runner."""

from __future__ import annotations

import time
import uuid
from typing import TYPE_CHECKING, Any, Dict, Optional

from fathom.core.context.manager import ContextManager
from fathom.core.execution.engine import ExecutionEngine
from fathom.schemas.configuration import FathomConfig
from fathom.schemas.results import ExplorationResult, IntentResult

if TYPE_CHECKING:
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
        config: Optional[FathomConfig] = None,
    ) -> None:
        """
        Initialize runner with all configured ports.
        
        Args:
            device: Device port for mobile device interactions
            llm: LLM port for language model interactions
            memory: Memory port for session state and cross-run memory
            knowledge: Knowledge port for application knowledge graph
            signal: Signal port for human-in-the-loop control
            storage: Storage port for artifact persistence
            telemetry: Telemetry port for observability
            config: Optional configuration (uses defaults if not provided)
        """
        self._device = device
        self._llm = llm
        self._memory = memory
        self._knowledge = knowledge
        self._signal = signal
        self._storage = storage
        self._telemetry = telemetry
        self._config = config or FathomConfig()
        
        # Wire core components
        self._engine = ExecutionEngine(
            device=device,
            llm=llm,
            memory=memory,
            signal=signal,
            storage=storage,
            telemetry=telemetry,
        )
        self._context_manager = ContextManager(memory=memory)
        
        # Track current workflow for cancellation
        self._current_strategy: Optional[object] = None

    async def run_intent(
        self,
        intent: str,
        max_steps: int = 20,
        use_xml: bool = False,
        request_id: Optional[str] = None,
        device_serial: Optional[str] = None,
        prompt_version: Optional[str] = None,
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
        
        Returns:
            IntentResult with execution outcome and metrics
        """
        workflow_id = request_id or uuid.uuid4().hex[:8]
        start_time = time.time()
        
        self._telemetry.info(
            "Starting intent workflow",
            intent=intent,
            max_steps=max_steps,
            workflow_id=workflow_id,
        )
        
        # Initialize context
        self._context_manager.set_roadmap(intent=intent)
        
        # Create and execute strategy
        from fathom.strategies.intent import IntentStrategy
        
        strategy = IntentStrategy(
            engine=self._engine,
            context=self._context_manager,
            intent=intent,
            device=self._device,
            llm=self._llm,
            memory=self._memory,
            storage=self._storage,
            telemetry=self._telemetry,
            max_steps=max_steps or self._config.intent_strategy.max_steps,
            use_xml=use_xml if use_xml is not None else self._config.intent_strategy.use_xml,
            workflow_id=workflow_id,
        )
        self._current_strategy = strategy
        
        try:
            # Execute strategy
            execution_result = await strategy.execute(max_steps=max_steps)
            
            # Get progress info
            progress = strategy.get_progress()
            
            # Collect metrics from strategy
            metrics = progress.get("metrics", {})
            
            # Get memory summary
            memory_summary = await self._get_memory_summary()
            
            # Build IntentResult
            duration = time.time() - start_time
            
            result = IntentResult(
                success=execution_result.success,
                completion_reason=execution_result.error or "Completed successfully" if execution_result.success else "Failed",
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
            
            self._telemetry.info(
                "Intent workflow completed",
                success=result.success,
                steps_taken=result.steps_taken,
                duration=duration,
            )
            
            return result
            
        finally:
            self._current_strategy = None

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
        
        self._telemetry.info(
            "Starting exploration workflow",
            max_steps=max_steps,
            workflow_id=workflow_id,
        )
        
        # Initialize context
        self._context_manager.set_roadmap(intent="Explore application structure")
        
        # Create and execute strategy
        from fathom.strategies.exploration import ExplorationStrategy
        
        strategy = ExplorationStrategy(
            engine=self._engine,
            context=self._context_manager,
            device=self._device,
            storage=self._storage,
            telemetry=self._telemetry,
            max_steps=max_steps or self._config.exploration_strategy.max_steps,
            timeout=self._config.exploration_strategy.timeout,
            seed=self._config.exploration_strategy.seed,
        )
        self._current_strategy = strategy
        
        try:
            # Execute strategy
            execution_result = await strategy.execute(max_steps=max_steps or self._config.exploration_strategy.max_steps)
            
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
            screen_graph = self._export_graph(graph)
            
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
            
            self._telemetry.info(
                "Exploration workflow completed",
                unique_screens=result.unique_screens,
                total_actions=result.total_actions,
                duration=duration,
            )
            
            return result
            
        finally:
            self._current_strategy = None

    def cancel(self) -> None:
        """Cancel the currently running workflow."""
        if self._current_strategy:
            self._telemetry.warning("Workflow cancellation requested")
            # TODO: Implement cancellation mechanism in strategies
    
    async def cleanup(self) -> None:
        """Cleanup resources."""
        # Cleanup LLM resources
        await self._llm.cleanup()
        
        self._telemetry.info("Runner cleanup completed")
    
    async def _get_memory_summary(self) -> Dict[str, Any]:
        """Get memory summary from memory port."""
        try:
            # Get all memory entries
            all_entries = await self._memory.get_all()
            
            # Count unique screens
            unique_screens = len({entry.get("visual_hash") for entry in all_entries if "visual_hash" in entry})
            
            # Count experiences
            experience_count = len([entry for entry in all_entries if entry.get("type") == "experience"])
            
            return {
                "total_entries": len(all_entries),
                "unique_screens": unique_screens,
                "experiences": experience_count,
            }
        except Exception as exception:
            self._telemetry.warning(f"Failed to get memory summary: {exception}")
            return {}
    
    def _export_graph(self, graph: ExplorationGraph) -> Dict[str, Any]:
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
            self._telemetry.warning(f"Failed to export graph: {exception}")
            return {}
    
    @property
    def engine(self) -> ExecutionEngine:
        """Get the execution engine."""
        return self._engine
    
    @property
    def context(self) -> ContextManager:
        """Get the context manager."""
        return self._context_manager
