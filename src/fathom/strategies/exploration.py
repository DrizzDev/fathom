from __future__ import annotations

import time
from logging import getLogger
from typing import Any, Dict, Optional

from fathom.base.paths import SharedPathManager
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

logger = getLogger(name=__name__)


class ExplorationStrategy:
    """
    Strategy for autonomous application mapping using LangGraph.
    """

    def __init__(
        self,
        llm: LLMPort,
        device: DevicePort,
        memory: MemoryPort,
        signal: SignalPort,
        storage: StoragePort,
        telemetry: TelemetryPort,
        path_manager: SharedPathManager,
        *,
        max_steps: int = 100,
        timeout: float = 3600.0,
        seed: Optional[int] = None,
        workflow_id: str = "exploration",
        package_name: str = "unknown_app",
    ) -> None:
        self.__graph_context = GraphContext(
            llm=llm,
            device=device,
            signal=signal,
            memory=memory,
            storage=storage,
            telemetry=telemetry,
            max_steps=max_steps,
            workflow_id=workflow_id,
            package_name=package_name,
            path_manager=path_manager,
            intent="Explore application",
        )

        self.__graph = build_exploration_graph(context=self.__graph_context)

    async def execute(self, max_steps: int) -> ExecutionResult:
        """
        Execute exploration.
        """

        _ = max_steps
        start_time = time.time()

        try:
            await self.__graph.ainvoke({})

            duration = int((time.time() - start_time) * 1000)
            return ExecutionResult(success=True, duration=duration)

        except Exception as exception:
            logger.exception(f"Exploration failed: {exception}")
            duration = int((time.time() - start_time) * 1000)
            return ExecutionResult(success=False, duration=duration, error=str(exception))

    @property
    def graph(self) -> ExplorationGraph:
        """
        Get the exploration graph.
        """

        return self.__graph_context.exploration_graph

    def get_progress(self) -> Dict[str, Any]:
        """
        Get progress.
        """

        return {
            "stats": self.graph.get_stats(),
            "steps": self.__graph_context.agent_state.step_count,
        }

    def cancel(self) -> None:
        """
        Cancel execution.
        """

        self.__graph_context.cancel()
