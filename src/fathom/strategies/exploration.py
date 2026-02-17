"""
Exploration-based execution strategy using LangGraph.
"""

from __future__ import annotations

import time
from logging import getLogger
from typing import TYPE_CHECKING, Any, Dict, Optional

from fathom.interfaces.device import DevicePort
from fathom.interfaces.llm import LLMPort
from fathom.interfaces.memory import MemoryPort
from fathom.interfaces.signal import SignalPort
from fathom.interfaces.storage import StoragePort
from fathom.interfaces.telemetry import TelemetryPort
from fathom.schemas.exploration import ExplorationGraph
from fathom.schemas.results import ExecutionResult
from fathom.strategies.graph.context import GraphContext
from fathom.strategies.graph.exploration.builder import build_exploration_graph

if TYPE_CHECKING:
    from fathom.base.paths import SharedPathManager

logger = getLogger(name=__name__)


class ExplorationStrategy:
    """
    Strategy for autonomous application mapping using LangGraph.
    """

    def __init__(
        self,
        *,
        device: DevicePort,
        llm: LLMPort,
        memory: MemoryPort,
        storage: StoragePort,
        telemetry: TelemetryPort,
        signal: SignalPort,
        path_manager: SharedPathManager,
        max_steps: int = 100,
        timeout: float = 3600.0,
        seed: Optional[int] = None,
        package_name: str = "unknown_app",
        workflow_id: str = "exploration",
    ) -> None:
        self.__graph_context = GraphContext(
            intent="Explore application",
            device=device,
            llm=llm,
            memory=memory,
            storage=storage,
            telemetry=telemetry,
            signal=signal,
            path_manager=path_manager,
            max_steps=max_steps,
            workflow_id=workflow_id,
            package_name=package_name,
        )

        self.__graph = build_exploration_graph(context=self.__graph_context)

    async def execute(self, max_steps: int) -> ExecutionResult:
        """Execute exploration."""
        start_time = time.time()

        try:
            await self.__graph.ainvoke({})

            duration = int((time.time() - start_time) * 1000)
            return ExecutionResult(
                success=True,
                duration=duration,
            )

        except Exception as exception:
            logger.exception(f"Exploration failed: {exception}")
            duration = int((time.time() - start_time) * 1000)
            return ExecutionResult(success=False, duration=duration, error=str(exception))

    @property
    def graph(self) -> ExplorationGraph:
        """Get the exploration graph."""
        return self.__graph_context.exploration_graph

    def get_progress(self) -> Dict[str, Any]:
        """Get progress."""
        return {
            "steps": self.__graph_context.agent_state.step_count,
            "stats": self.graph.get_stats(),
        }

    def cancel(self) -> None:
        """Cancel execution."""
        self.__graph_context.cancel()
