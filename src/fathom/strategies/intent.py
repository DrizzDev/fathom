from __future__ import annotations

import importlib
import time
from contextlib import asynccontextmanager
from logging import getLogger
from pathlib import Path  # noqa: TC003
from typing import Any, AsyncIterator, Dict, List, Optional, cast

from langgraph.checkpoint.memory import MemorySaver
from rich.console import Console

from fathom.base.paths import SharedPathManager
from fathom.constants.events import FathomEvent
from fathom.constants.graph import NodeName
from fathom.constants.state import CommonStateKey, IntentStateKey
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
        self.__llm = llm
        self.__step_results: List[StepResult] = []
        self.__graph = None

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

        interrupt_nodes = [] if not signal.is_interactive else [NodeName.EXECUTE.value]

        # Compatibility: newer langgraph JsonPlusSerializer() has no
        # allowed_json_modules argument, while older versions do.

        # Defer checkpointer + graph construction to execute(), because SqliteSaver is a
        # context manager and must stay open for the duration of the graph run.
        self.__graph_builder = builder
        self.__interrupt_nodes = interrupt_nodes
        self.__checkpoint_db = path_manager.memory_path / "checkpoints.db"

    async def execute(self) -> ExecutionResult:
        """
        Execute intent-based workflow via specialized executor.
        """

        from fathom.runtime.executor import GraphExecutor

        start_time = time.time()

        try:
            async with self.__build_checkpointer_context(
                checkpoint_db_path=self.__checkpoint_db
            ) as checkpointer:
                self.__graph = self.__graph_builder.build(
                    checkpointer=checkpointer,
                    interrupt_before=self.__interrupt_nodes,
                )
                # 1. Decompose intent into sub-goals using LLM
                logger.info(f"[IntentStrategy] Decomposing intent: {self.__intent}")
                decomposer = IntentDecomposer.with_configuration(
                    llm=self.__llm, configuration=self.__graph_context.configuration.llm
                )
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
                    has_interrupts=self.__graph_context.signal.is_interactive,
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

                if self.__graph is None:
                    raise RuntimeError("Intent graph is not initialized")

                config = cast("RunnableConfig", {"configurable": {"thread_id": self.__workflow_id}})
                final_state = await self.__graph.aget_state(config)

            is_cancelled = self.__graph_context.is_cancelled
            success = self.__graph_context.agent_state.is_complete
            error = final_state.values.get(CommonStateKey.COMPLETION_REASON)
            if not error:
                error = final_state.values.get("completion_reason")
            if not error:
                error = self.__graph_context.agent_state.completion_reason
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
                if self.__graph is not None:
                    final_state = await self.__graph.aget_state(config)
                else:
                    final_state = None
                self.__step_results = list(
                    (final_state.values.get(IntentStateKey.STEP_RESULTS) if final_state else [])
                    or []
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
        # SubGoalStatus.SKIPPED was removed; callers still expect a skipped list.
        skipped: list[str] = []

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

    @asynccontextmanager
    async def __build_checkpointer_context(
        self,
        checkpoint_db_path: Path,
    ) -> AsyncIterator[Any]:
        """
        Build a persistence layer for graph checkpoints as an async context manager.

        Prefers AsyncSqliteSaver for crash-safe persistence.
        Falls back to in-memory checkpoints if sqlite support is unavailable.
        """

        try:
            checkpoint_db_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exception:
            logger.error(
                "Failed to create checkpoint directory for SQLite checkpointer. "
                f"checkpoint_db_path={checkpoint_db_path}. Reason: {exception}"
            )
            raise

        try:
            # Use importlib to avoid static import resolution errors.
            aio_module = importlib.import_module("langgraph.checkpoint.sqlite.aio")
        except (ImportError, ModuleNotFoundError) as exception:
            logger.warning(
                "AsyncSqliteSaver unavailable; falling back to MemorySaver. "
                "Install 'langgraph-checkpoint-sqlite' and 'aiosqlite' to enable "
                f"persistent checkpoints. Reason: {exception}"
            )
            yield MemorySaver()
            return

        AsyncSqliteSaver = aio_module.AsyncSqliteSaver

        try:
            async with AsyncSqliteSaver.from_conn_string(str(checkpoint_db_path)) as checkpointer:
                logger.info(f"Using AsyncSqliteSaver for checkpointing at {checkpoint_db_path}")
                yield checkpointer
        except Exception as exception:
            logger.error(
                "Failed to initialize AsyncSqliteSaver. "
                f"checkpoint_db_path={checkpoint_db_path}. Reason: {exception}"
            )
            raise
