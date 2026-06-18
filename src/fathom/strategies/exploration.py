from __future__ import annotations

import asyncio
import time
from logging import getLogger
from typing import TYPE_CHECKING, Any, Dict, Optional

from fathom.base.paths import SharedPathManager
from fathom.constants.exploration import DEFAULT_EXPLORATION_INTENT, DISABLED_LOOP_THRESHOLD
from fathom.core.agent.state import AgentState
from fathom.core.services.telemetry import PhaseAnnouncer
from fathom.infrastructure.memory.knowledge_graph import KnowledgeGraph
from fathom.interfaces.device import DevicePort
from fathom.interfaces.llm import LLMPort
from fathom.interfaces.memory import MemoryPort
from fathom.interfaces.perception import PerceptionPort
from fathom.interfaces.signal import SignalPort
from fathom.interfaces.storage import StoragePort
from fathom.interfaces.telemetry import TelemetryPort
from fathom.schemas.configuration import FathomConfiguration
from fathom.schemas.results import ExecutionResult
from fathom.strategies.graph.context import GraphContext
from fathom.strategies.graph.exploration.builder import ExplorationGraphBuilder

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig

    from fathom.core.config.loader import RuntimeConfigLoader

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
        runtime_configuration: Optional["RuntimeConfigLoader"] = None,
        intent: Optional[str] = None,
        focus: Optional[str] = None,
    ) -> None:
        self.__seed = seed
        self.__timeout = timeout
        self.__max_steps = max_steps
        effective_intent = intent or DEFAULT_EXPLORATION_INTENT

        # Exploration strategy doesn't use XML grounding (uses visual-only approach)
        from fathom.core.config.loader import RuntimeConfigLoader

        # Use the caller-bound loader when supplied; otherwise fall
        # back to env-only construction so stand-alone / test paths
        # remain unchanged.
        loader = (
            runtime_configuration if runtime_configuration is not None else RuntimeConfigLoader()
        )

        phase = PhaseAnnouncer(
            telemetry=telemetry,
            message=configuration.telemetry.phase,
        )

        self.__graph_context = GraphContext(
            llm=llm,
            phase=phase,
            intent=effective_intent,
            focus=focus,
            use_xml=False,
            device=device,
            signal=signal,
            memory=memory,
            storage=storage,
            telemetry=telemetry,
            max_steps=max_steps,
            perception=perception,
            workflow_id=workflow_id,
            package_name=package_name,
            path_manager=path_manager,
            configuration=configuration,
            perception_configuration=loader.perception(),
        )

        # DFS exploration revisits screens by design, so loop detection (which
        # would otherwise flag the agent as stuck) is disabled for the run. The
        # graph context supplies the live runtime capabilities for the agent.
        self.__graph_context.set_agent_state(
            AgentState(
                intent=effective_intent,
                max_steps=max_steps,
                capabilities=self.__graph_context.capabilities,
                loop_threshold=DISABLED_LOOP_THRESHOLD,
            )
        )

        builder = ExplorationGraphBuilder(context=self.__graph_context)
        self.__graph = builder.build()

    async def execute(self) -> ExecutionResult:
        """
        Execute exploration with timeout.
        """

        start_time = time.time()

        try:
            # Hydrate cross-run knowledge before exploring.
            await self.__graph_context.exploration_graph.load()

            # Each exploration step walks several nodes (ground -> route -> scan
            # -> execute -> record), so the LangGraph super-step budget must scale
            # with max_steps -- the default of 25 would abort almost immediately.
            config: RunnableConfig = {"recursion_limit": self.__max_steps * 10 + 100}

            # Execute with timeout if configured
            if self.__timeout > 0:
                await asyncio.wait_for(
                    self.__graph.ainvoke({}, config=config), timeout=self.__timeout
                )
            else:
                await self.__graph.ainvoke({}, config=config)

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
    def graph(self) -> KnowledgeGraph:
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
