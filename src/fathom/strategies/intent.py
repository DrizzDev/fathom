from __future__ import annotations

import importlib
import time
from logging import getLogger
from pathlib import Path  # noqa: TC003
from typing import Any, Dict, Optional, cast

from langgraph.checkpoint.memory import MemorySaver
from rich.console import Console

from fathom.adapters.signal.noop import NoopSignal
from fathom.base.paths import SharedPathManager
from fathom.constants.events import FathomEvent
from fathom.constants.graph import NodeName
from fathom.constants.state import CommonStateKey
from fathom.core.services.decomposer import IntentDecomposer
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
        self.__llm = llm

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

        # Compatibility: newer langgraph JsonPlusSerializer() has no
        # allowed_json_modules argument, while older versions do.

        # Build checkpointer for graph state persistence
        checkpoint_db = path_manager.memory_path / "checkpoints.db"
        checkpointer = self.__build_checkpointer(checkpoint_db_path=checkpoint_db)

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
            # 1. Decompose intent into sub-goals using LLM
            logger.info(f"[IntentStrategy] Decomposing intent: {self.__intent}")
            decomposer = IntentDecomposer(llm=self.__llm)
            sub_goals = await decomposer.decompose(intent=self.__intent)

            # Set sub-goals in agent state
            self.__graph_context.agent_state.set_sub_goals(sub_goals)
            logger.info(
                f"[IntentStrategy] Intent decomposed into {len(sub_goals)} sub-goals. "
                f"Starting execution..."
            )

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
                    "Final script generation returned empty data; cannot publish SCRIPT_GENERATED event"
                )

            # 3. Result extraction from final state
            from langchain_core.runnables.config import RunnableConfig

            config = cast("RunnableConfig", {"configurable": {"thread_id": self.__workflow_id}})
            final_state = await self.__graph.aget_state(config)

            is_cancelled = self.__graph_context.is_cancelled
            success = self.__graph_context.agent_state.is_complete
            error = final_state.values.get(CommonStateKey.COMPLETION_REASON)
            if not error:
                error = final_state.values.get("completion_reason")
            if not error:
                error = self.__graph_context.agent_state.completion_reason

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
            return ExecutionResult(
                success=False, duration=duration, error=str(exception), is_cancelled=is_cancelled
            )

    def get_progress(self) -> Dict[str, Any]:
        """
        Get execution progress.
        """

        return {
            "intent": self.__intent,
            "step_count": self.__graph_context.agent_state.step_count,
            "is_complete": self.__graph_context.agent_state.is_complete,
            "completion_reason": self.__graph_context.agent_state.completion_reason,
        }

    def get_subgoal_execution_audit(self) -> tuple[list[str], list[str], int]:
        """
        Get audit trail of executed vs skipped subgoals.

        Returns:
            Tuple of (executed_descriptions, skipped_descriptions, total_count)
        """
        from fathom.schemas.subgoal import SubGoalStatus

        subgoals = self.__graph_context.agent_state.sub_goal_list
        executed = [sg.description for sg in subgoals if sg.status == SubGoalStatus.COMPLETE]
        skipped = [sg.description for sg in subgoals if sg.status == SubGoalStatus.SKIPPED]

        return executed, skipped, len(subgoals)

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

    def __build_checkpointer(
        self,
        checkpoint_db_path: Path,
    ) -> Any:
        """
        Build a persistence layer for graph checkpoints.

        Prefers SQLite-backed checkpoints for crash-safe persistence.
        Falls back to in-memory checkpoints if sqlite support is unavailable.
        """

        checkpoint_db_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Use importlib to avoid static import resolution errors
            sqlite_module = importlib.import_module("langgraph.checkpoint.sqlite")
            SqliteSaver = sqlite_module.SqliteSaver

            # Create checkpointer using SqliteSaver
            # The default serializer in SqliteSaver handles Pydantic models automatically
            checkpointer = SqliteSaver.from_conn_string(str(checkpoint_db_path))
            logger.info(f"Using SqliteSaver for checkpointing at {checkpoint_db_path}")
            return checkpointer
        except Exception as exception:
            logger.warning(
                "SQLite checkpoint saver unavailable; falling back to MemorySaver. "
                "Install 'langgraph-checkpoint-sqlite' to enable persistent checkpoints. "
                f"Reason: {exception}"
            )
            return MemorySaver()
