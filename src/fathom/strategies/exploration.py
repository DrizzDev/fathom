from __future__ import annotations

import asyncio
import time
from logging import getLogger
from typing import Any, Dict, Optional

from fathom.base.paths import SharedPathManager
from fathom.interfaces.device import DevicePort
from fathom.interfaces.llm import LLMPort
from fathom.interfaces.memory import MemoryPort
from fathom.interfaces.perception import PerceptionPort
from fathom.interfaces.signal import SignalPort
from fathom.interfaces.storage import StoragePort
from fathom.interfaces.telemetry import TelemetryPort
from fathom.schemas.configuration import FathomConfiguration
from fathom.schemas.exploration import ExplorationGraph
from fathom.schemas.results import ExecutionResult
from fathom.strategies.graph.context import GraphContext
from fathom.strategies.graph.exploration.builder import ExplorationGraphBuilder

logger = getLogger(name=__name__)


class ExplorationStrategy:
    """
    Strategy for autonomous application mapping using LangGraph.
    """

    def __init__(
        self,
        max_steps: int,
        timeout: float,
        workflow_id: str,
        package_name: str,
        seed: Optional[int],
        *,
        llm: LLMPort,
        device: DevicePort,
        perception: PerceptionPort,
        memory: MemoryPort,
        signal: SignalPort,
        storage: StoragePort,
        telemetry: TelemetryPort,
        path_manager: SharedPathManager,
        configuration: FathomConfiguration,
    ) -> None:
        self.__seed = seed
        self.__timeout = timeout
        intent = "Explore application"

        # Exploration strategy doesn't use XML grounding (uses visual-only approach)
        self.__graph_context = GraphContext(
            llm=llm,
            intent=intent,
            use_xml=False,
            device=device,
            perception=perception,
            signal=signal,
            memory=memory,
            storage=storage,
            telemetry=telemetry,
            max_steps=max_steps,
            workflow_id=workflow_id,
            package_name=package_name,
            path_manager=path_manager,
            configuration=configuration,
        )

        builder = ExplorationGraphBuilder(context=self.__graph_context)
        self.__graph = builder.build()

    async def execute(self) -> ExecutionResult:
        """
        Execute exploration with timeout.
        """

        start_time = time.time()

        try:
            # Execute with timeout if configured
            if self.__timeout > 0:
                await asyncio.wait_for(self.__graph.ainvoke({}), timeout=self.__timeout)
            else:
                await self.__graph.ainvoke({})

            duration = int((time.time() - start_time) * 1000)
            return ExecutionResult(success=True, duration=duration)

        except asyncio.TimeoutError:
            logger.warning(f"Exploration timed out after {self.__timeout}s")
            duration = int((time.time() - start_time) * 1000)
            return ExecutionResult(
                success=False, duration=duration, error=f"Timeout after {self.__timeout}s"
            )
        except Exception as exception:
            logger.exception(f"Exploration failed: {exception}")
            duration = int((time.time() - start_time) * 1000)
            return ExecutionResult(success=False, duration=duration, error=str(exception))
        finally:
            try:
                await self.__graph_context.shutdown()
            except Exception as shutdown_error:
                logger.warning(
                    f"[exploration-strategy] graph context shutdown failed: {shutdown_error}"
                )

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
