from __future__ import annotations

import time
from logging import getLogger
from typing import Any, Dict, Optional

from langgraph.checkpoint.memory import MemorySaver
from rich.console import Console

from fathom.base.paths import SharedPathManager
from fathom.constants.graph import NodeName
from fathom.interfaces.device import DevicePort
from fathom.interfaces.llm import LLMPort
from fathom.interfaces.memory import MemoryPort
from fathom.interfaces.signal import SignalPort
from fathom.interfaces.storage import StoragePort
from fathom.interfaces.summarization import SummarizationPort
from fathom.interfaces.telemetry import TelemetryPort
from fathom.runtime.executor import GraphExecutor
from fathom.schemas.metrics import ExecutionMetrics
from fathom.schemas.orchestration import RealignmentPolicy
from fathom.schemas.results import ExecutionResult
from fathom.strategies.graph.context import GraphContext
from fathom.strategies.graph.intent.builder import IntentGraphBuilder

console = Console()
logger = getLogger(name=__name__)


class IntentStrategy:
    """
    Strategy for executing a specific intent using LangGraph.
    """

    def __init__(
        self,
        intent: str,
        llm: LLMPort,
        device: DevicePort,
        memory: MemoryPort,
        signal: SignalPort,
        storage: StoragePort,
        telemetry: TelemetryPort,
        summarizer: SummarizationPort,
        path_manager: SharedPathManager,
        *,
        max_steps: int = 20,
        use_xml: bool = False,
        workflow_id: str = "default",
        package_name: str = "unknown_app",
        realignment: Optional[RealignmentPolicy] = None,
    ) -> None:
        self.__intent = intent
        self.__workflow_id = workflow_id

        # Initialize Graph Context with injected summarizer
        self.__graph_context = GraphContext(
            llm=llm,
            intent=intent,
            device=device,
            memory=memory,
            signal=signal,
            use_xml=use_xml,
            storage=storage,
            max_steps=max_steps,
            telemetry=telemetry,
            summarizer=summarizer,
            workflow_id=workflow_id,
            realignment=realignment,
            path_manager=path_manager,
            package_name=package_name,
        )

        # 1. Build Graph with Interrupts (Injected dependency: MemorySaver)
        builder = IntentGraphBuilder(context=self.__graph_context)

        # We interrupt before critical decision points to allow HITL via Strategy Loop
        self.__graph = builder.build(
            checkpointer=MemorySaver(),
            interrupt_before=[NodeName.ANALYZE, NodeName.EXECUTE],
        )

    async def execute(self, max_steps: int) -> ExecutionResult:
        """
        Execute intent-based workflow via specialized executor.
        """

        _ = max_steps
        start_time = time.time()

        try:
            # 2. Delegate execution lifecycle to the GraphExecutor (SRP)
            # invalidate_on_injection=True forces re-planning when context is added
            executor = GraphExecutor(
                graph=self.__graph,
                context=self.__graph_context,
                thread_id=self.__workflow_id,
                invalidate_on_injection=self.__graph_context.realignment.immediate,
            )

            await executor.run()

            # 3. Result extraction from final state
            config = {"configurable": {"thread_id": self.__workflow_id}}
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

    def get_progress(self) -> Dict[str, Any]:
        """
        Get execution progress.
        """

        return {
            "intent": self.__intent,
            "step_count": self.__graph_context.agent_state.step_count,
            "is_complete": self.__graph_context.agent_state.is_complete,
        }

    def get_metrics(self) -> ExecutionMetrics:
        """
        Get execution metrics.
        """

        return self.__graph_context.metrics

    def cancel(self) -> None:
        """
        Cancel the execution.
        """

        self.__graph_context.cancel()
