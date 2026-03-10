from __future__ import annotations

import time
from logging import getLogger
from typing import Any, Dict, List, Optional

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from rich.console import Console

from fathom.adapters.signal.noop import NoopSignal
from fathom.base.paths import SharedPathManager
from fathom.constants.events import FathomEvent
from fathom.constants.graph import NodeName
from fathom.constants.state import IntentStateKey
from fathom.interfaces.device import DevicePort
from fathom.interfaces.llm import LLMPort
from fathom.interfaces.memory import MemoryPort
from fathom.interfaces.signal import SignalPort
from fathom.interfaces.storage import StoragePort
from fathom.interfaces.summarization import SummarizationPort
from fathom.interfaces.telemetry import TelemetryPort
from fathom.schemas.configuration import FathomConfiguration
from fathom.schemas.metrics import ExecutionMetrics
from fathom.schemas.orchestration import RealignmentPolicy
from fathom.schemas.results import ExecutionResult
from fathom.schemas.steps import StepResult
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
        configuration: FathomConfiguration,
        *,
        use_xml: bool,
        max_steps: int,
        workflow_id: str,
        package_name: str,
        realignment: Optional[RealignmentPolicy] = None,
    ) -> None:
        self.__intent = intent
        self.__workflow_id = workflow_id
        self.__step_results: List[StepResult] = []

        # Initialize Graph Context with injected summarizer
        self.__graph_context = GraphContext(
            llm=llm,
            intent=intent,
            device=device,
            memory=memory,
            signal=signal,
            use_xml=use_xml,
            storage=storage,
            telemetry=telemetry,
            max_steps=max_steps,
            summarizer=summarizer,
            workflow_id=workflow_id,
            realignment=realignment,
            package_name=package_name,
            path_manager=path_manager,
            configuration=configuration,
        )

        # 1. Build Graph with Interrupts (Injected dependency: MemorySaver)
        builder = IntentGraphBuilder(context=self.__graph_context)

        # Use checkpointer only for interactive mode (with interrupts)
        # Autonomous mode doesn't need checkpointing

        interrupt_nodes = [] if isinstance(signal, NoopSignal) else [NodeName.EXECUTE.value]

        # Configure serializer to allow fathom and langgraph modules for MsgPack
        serializer = JsonPlusSerializer(
            allowed_json_modules=[
                "fathom",
                "fathom.constants",
                "fathom.constants.state",
                "fathom.schemas.screens",
                "fathom.schemas.steps",
                "fathom.schemas.results",
                "langgraph",
            ]
        )
        checkpointer = MemorySaver(serde=serializer)

        self.__graph = builder.build(
            checkpointer=checkpointer,
            interrupt_before=interrupt_nodes,
        )

    async def execute(self) -> ExecutionResult:
        """
        Execute intent-based workflow via specialized executor.
        """

        from fathom.runtime.executor import GraphExecutor

        start_time = time.time()

        try:
            # 2. Delegate execution lifecycle to the GraphExecutor (SRP)
            # invalidate_on_injection=True forces re-planning when context is added
            executor = GraphExecutor(
                graph=self.__graph,
                context=self.__graph_context,
                thread_id=self.__workflow_id,
                invalidate_on_injection=self.__graph_context.realignment.immediate,
                has_interrupts=not isinstance(self.__graph_context.signal, NoopSignal),
            )

            await executor.run()

            if self.__graph_context.is_cancelled:
                script_data = await self.__graph_context.history.get_current_script(
                    intent=self.__intent
                )
                if script_data:
                    await self.__graph_context.telemetry.info(
                        script_data,
                        type=FathomEvent.SCRIPT_GENERATED,
                        step=self.__graph_context.agent_state.step_count,
                    )
                else:
                    logger.warning(
                        "Execution is cancelled and script data is missing. Cannot publish SCRIPT_GENERATED event"
                    )

            # 3. Result extraction from final state
            config = {"configurable": {"thread_id": self.__workflow_id}}
            final_state = await self.__graph.aget_state(config)

            is_cancelled = self.__graph_context.is_cancelled
            success = self.__graph_context.agent_state.is_complete
            error = final_state.values.get("completion_reason")
            self.__step_results = list(final_state.values.get(IntentStateKey.STEP_RESULTS) or [])

            duration = int((time.time() - start_time) * 1000)

            return ExecutionResult(
                duration=duration,
                is_cancelled=is_cancelled,
                success=success and not is_cancelled,
                error=error if not success else None,
            )

        except Exception as exception:
            logger.exception(f"Intent strategy execution failed: {exception}")
            duration = int((time.time() - start_time) * 1000)
            is_cancelled = self.__graph_context.is_cancelled

            # Recover step history from last checkpoint so the execution transcript
            # is not lost even when the run raises an exception.
            try:
                config = {"configurable": {"thread_id": self.__workflow_id}}
                final_state = await self.__graph.aget_state(config)
                self.__step_results = list(
                    final_state.values.get(IntentStateKey.STEP_RESULTS) or []
                )
            except Exception as recovery_error:
                logger.debug(f"Could not recover step results from checkpoint: {recovery_error}")

            return ExecutionResult(
                success=False, duration=duration, error=str(exception), is_cancelled=is_cancelled
            )

    @property
    def step_results(self) -> List[StepResult]:
        """
        Step results accumulated during execution.
        Available after execute() completes.
        """

        return self.__step_results

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
