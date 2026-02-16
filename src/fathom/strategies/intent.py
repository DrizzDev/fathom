"""
Intent-based execution strategy using LangGraph.
"""

from __future__ import annotations

import time
from logging import getLogger
from typing import TYPE_CHECKING, Optional

from rich.console import Console

from fathom.interfaces.device import DevicePort
from fathom.interfaces.llm import LLMPort
from fathom.interfaces.memory import MemoryPort
from fathom.interfaces.signal import SignalPort
from fathom.interfaces.storage import StoragePort
from fathom.interfaces.telemetry import TelemetryPort
from fathom.schemas.results import ExecutionResult
from fathom.strategies.graph.context import GraphContext
from fathom.strategies.graph.intent.builder import build_intent_graph

if TYPE_CHECKING:
    from fathom.base.paths import SharedPathManager

logger = getLogger(name=__name__)
console = Console()


class IntentStrategy:
    """
    Strategy for executing a specific intent using LangGraph.
    """

    def __init__(
        self,
        intent: str,
        *,
        device: DevicePort,
        llm: LLMPort,
        memory: MemoryPort,
        storage: StoragePort,
        telemetry: TelemetryPort,
        signal: SignalPort,
        path_manager: SharedPathManager,
        max_steps: int = 20,
        use_xml: bool = False,
        workflow_id: str = "default",
        package_name: str = "unknown_app",
    ) -> None:
        self.__intent = intent
        self.__workflow_id = workflow_id
        
        # Initialize Graph Context
        self.__graph_context = GraphContext(
            intent=intent,
            device=device,
            llm=llm,
            memory=memory,
            storage=storage,
            telemetry=telemetry,
            signal=signal,
            path_manager=path_manager,
            max_steps=max_steps,
            use_xml=use_xml,
            workflow_id=workflow_id,
            package_name=package_name,
        )
        
        self.__graph = build_intent_graph(context=self.__graph_context)

    async def execute(self, max_steps: int) -> ExecutionResult:
        """Execute intent-based workflow."""
        start_time = time.time()
        
        # Configuration for checkpointing
        config = {"configurable": {"thread_id": self.__workflow_id}}
        
        try:
            # Stream execution to allow HITL intervention between nodes
            async for event in self.__graph.astream({}, config=config):
                
                # Check for control signals (Pause/Interrupt)
                signal_type = await self.__graph_context.signal.check_signal()
                
                if signal_type:
                    # Block execution until user resumes
                    await self.__graph_context.signal.wait_for_resume()
                    
                    # Check if user injected new context/instruction
                    if hasattr(self.__graph_context.signal, "get_injected_context"):
                        injected = self.__graph_context.signal.get_injected_context()
                        if injected:
                            logger.info(f"Injecting user context: {injected}")
                            # Update graph state dynamically
                            await self.__graph.aupdate_state(
                                config, 
                                {"injected_context": injected}
                            )
                
                # Check cancellation
                if self.__graph_context.is_cancelled:
                    logger.warning("Graph execution cancelled by user")
                    break

            # Result extraction from final state
            final_state = await self.__graph.aget_state(config)
            success = self.__graph_context.agent_state.is_complete
            error = final_state.values.get("completion_reason")
            
            duration = int((time.time() - start_time) * 1000)
            
            return ExecutionResult(
                success=success,
                duration=duration,
                error=error if not success else None,
            )

        except Exception as exception:
            logger.exception(f"Intent strategy execution failed: {exception}")
            duration = int((time.time() - start_time) * 1000)
            return ExecutionResult(
                success=False,
                duration=duration,
                error=str(exception),
            )

    def get_progress(self) -> dict:
        """Get execution progress."""
        return {
            "intent": self.__intent,
            "step_count": self.__graph_context.agent_state.step_count,
            "is_complete": self.__graph_context.agent_state.is_complete,
        }

    def get_metrics(self) -> object:
        """Get execution metrics."""
        return self.__graph_context.metrics

    def cancel(self) -> None:
        """Cancel the execution."""
        self.__graph_context.cancel()
