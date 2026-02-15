"""Fathom execution runner."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

from fathom.core.context.manager import ContextManager
from fathom.core.execution.engine import ExecutionEngine

if TYPE_CHECKING:
    from fathom.interfaces.device import DevicePort
    from fathom.interfaces.knowledge import KnowledgePort
    from fathom.interfaces.llm import LLMPort
    from fathom.interfaces.memory import MemoryPort
    from fathom.interfaces.signal import SignalPort
    from fathom.interfaces.storage import StoragePort
    from fathom.interfaces.telemetry import TelemetryPort


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
        """
        self._device = device
        self._llm = llm
        self._memory = memory
        self._knowledge = knowledge
        self._signal = signal
        self._storage = storage
        self._telemetry = telemetry
        
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

    async def run(
        self,
        *,
        intent: str,
        max_steps: int = 20,
        strategy: str = "intent",
    ) -> Dict[str, Any]:
        """
        Execute workflow with given intent.
        
        Args:
            intent: User intent to accomplish
            max_steps: Maximum execution steps
            strategy: Execution strategy ("intent" or "exploration")
        
        Returns:
            Execution result with outcome and metrics
        
        NOTE: This implementation provides the core execution infrastructure.
        Strategy implementations (IntentStrategy, ExplorationStrategy) will be
        added in Task 13 to provide the full workflow logic.
        """
        self._telemetry.info(
            "Starting execution",
            intent=intent,
            max_steps=max_steps,
            strategy=strategy,
        )
        
        # Initialize context with intent
        self._context_manager.set_roadmap(intent=intent)
        
        # For now, return a placeholder result
        # Full strategy implementation will be added in Task 13
        self._telemetry.info(
            "Execution infrastructure ready",
            message="Strategy implementations coming in Task 13",
        )
        
        return {
            "success": True,
            "steps": 0,
            "message": "Execution engine and context manager initialized. Strategy implementations coming in Task 13.",
            "intent": intent,
            "max_steps": max_steps,
            "strategy": strategy,
        }
    
    @property
    def engine(self) -> ExecutionEngine:
        """Get the execution engine."""
        return self._engine
    
    @property
    def context(self) -> ContextManager:
        """Get the context manager."""
        return self._context_manager
