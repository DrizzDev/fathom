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
        """
        self._telemetry.info(
            "Starting execution",
            intent=intent,
            max_steps=max_steps,
            strategy=strategy,
        )
        
        # Initialize context with intent
        self._context_manager.set_roadmap(intent=intent)
        
        # Select and execute strategy
        if strategy == "intent":
            from fathom.strategies.intent import IntentStrategy
            strategy_impl = IntentStrategy(
                engine=self._engine,
                context=self._context_manager,
                intent=intent,
                device=self._device,
                llm=self._llm,
                memory=self._memory,
                storage=self._storage,
                telemetry=self._telemetry,
                max_steps=max_steps,
            )
        elif strategy == "exploration":
            from fathom.strategies.exploration import ExplorationStrategy
            strategy_impl = ExplorationStrategy(
                engine=self._engine,
                context=self._context_manager,
                device=self._device,
                storage=self._storage,
                telemetry=self._telemetry,
                max_steps=max_steps,
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # Execute strategy
        result = await strategy_impl.execute(max_steps=max_steps)
        
        self._telemetry.info(
            "Execution completed",
            success=result.success,
            duration=result.duration,
        )
        
        return {
            "success": result.success,
            "duration": result.duration,
            "error": result.error,
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
