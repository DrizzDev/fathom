from __future__ import annotations

import importlib
import time
from contextlib import ExitStack, contextmanager
from logging import getLogger
from pathlib import Path  # noqa: TC003
from typing import Any, Dict, Iterator, List, Optional

from langgraph.checkpoint.memory import MemorySaver

from fathom.base.paths import SharedPathManager
from fathom.constants.events import FathomEvent
from fathom.constants.graph import NodeName
from fathom.constants.state import CompletionReason, IntentStateKey
from fathom.core.services.decomposer import IntentDecomposer
from fathom.interfaces.device import DevicePort
from fathom.interfaces.llm import LLMPort
from fathom.interfaces.memory import MemoryPort
from fathom.interfaces.perception import PerceptionPort
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
        perception: PerceptionPort,
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
        self.__graph: Any = None
        self.__completion_reason: Optional[str] = None

        self.__graph_context = GraphContext(
            llm=llm,
            intent=intent,
            device=device,
            perception=perception,
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

        builder = IntentGraphBuilder(context=self.__graph_context)
        interrupt_nodes = (
            [] if not signal.supports_interruption() else [NodeName.EXECUTE.value]
        )

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
            with ExitStack() as stack:
                checkpointer: Any = stack.enter_context(
                    self.__build_checkpointer_context(checkpoint_db_path=self.__checkpoint_db)
                )
                self.__graph = self.__graph_builder.build(
                    checkpointer=checkpointer,
                    interrupt_before=self.__interrupt_nodes,
                )

                logger.info(f"[IntentStrategy] Decomposing intent: {self.__intent}")
                decomposer = IntentDecomposer.with_configuration(
                    llm=self.__llm,
                    configuration=self.__graph_context.configuration.llm,
                )
                sub_goals = await decomposer.decompose(intent=self.__intent)

                self.__graph_context.agent_state.set_sub_goals(sub_goals)
                logger.info(
                    f"[IntentStrategy] Intent decomposed into {len(sub_goals)} sub-goals. "
                    "Starting execution..."
                )

                executor = GraphExecutor(
                    graph=self.__graph,
                    context=self.__graph_context,
                    thread_id=self.__workflow_id,
                    invalidate_on_injection=self.__graph_context.realignment.immediate,
                    has_interrupts=self.__graph_context.signal.supports_interruption(),
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
                        "Final script generation returned empty data; cannot publish "
                        "SCRIPT_GENERATED event"
                    )

                if self.__graph is None:
                    raise RuntimeError("Intent graph is not initialized")

                config = {"configurable": {"thread_id": self.__workflow_id}}
                final_state = await self.__graph.aget_state(config)

            is_cancelled = self.__graph_context.is_cancelled
            completion_reason = final_state.values.get("completion_reason")
            self.__step_results = list(final_state.values.get(IntentStateKey.STEP_RESULTS) or [])

            if completion_reason is None:
                completion_reason = self.__graph_context.agent_state.completion_reason

            self.__completion_reason = completion_reason
            success = self.__is_successful_completion(
                is_complete=self.__graph_context.agent_state.is_complete,
                is_cancelled=is_cancelled,
                completion_reason=completion_reason,
            )

            error = completion_reason if not success else None
            duration = int((time.time() - start_time) * 1000)

            return ExecutionResult(
                duration=duration,
                is_cancelled=is_cancelled,
                success=success and not is_cancelled,
                error=error,
            )

        except Exception as exception:
            logger.exception(f"Intent strategy execution failed: {exception}")
            duration = int((time.time() - start_time) * 1000)
            is_cancelled = self.__graph_context.is_cancelled

            try:
                config = {"configurable": {"thread_id": self.__workflow_id}}
                final_state = await self.__graph.aget_state(config) if self.__graph else None
                self.__step_results = list(
                    (final_state.values.get(IntentStateKey.STEP_RESULTS) if final_state else [])
                    or []
                )
            except Exception as recovery_error:
                logger.debug(f"Could not recover step results from checkpoint: {recovery_error}")

            return ExecutionResult(
                success=False,
                duration=duration,
                error=str(exception),
                is_cancelled=is_cancelled,
            )

    def __is_successful_completion(
        self,
        *,
        is_complete: bool,
        is_cancelled: bool,
        completion_reason: Optional[str],
    ) -> bool:
        """
        Determine whether the final completion state represents a successful outcome.
        """

        if not is_complete or is_cancelled:
            return False

        return completion_reason not in {
            None,
            CompletionReason.FAILED.value,
            CompletionReason.CANCELLED.value,
            CompletionReason.MAX_STEPS.value,
            CompletionReason.STUCK.value,
            CompletionReason.INTERVENTION_REQUIRED.value,
        }

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
        """

        from fathom.schemas.subgoal import SubGoalStatus

        subgoals = self.__graph_context.agent_state.sub_goal_list
        executed = [sg.description for sg in subgoals if sg.status == SubGoalStatus.COMPLETE]
        skipped: list[str] = []
        return executed, skipped, len(subgoals)

    @property
    def completion_reason(self) -> Optional[str]:
        """
        Return the final workflow completion reason.
        """

        return self.__completion_reason

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

    @contextmanager
    def __build_checkpointer_context(
        self,
        checkpoint_db_path: Path,
    ) -> Iterator[Any]:
        """
        Build a persistence layer for graph checkpoints as a context manager.

        Prefers SQLite-backed checkpoints for crash-safe persistence.
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
            sqlite_module = importlib.import_module("langgraph.checkpoint.sqlite")
        except (ImportError, ModuleNotFoundError) as exception:
            logger.warning(
                "SQLite checkpoint saver unavailable; falling back to MemorySaver. "
                "Install 'langgraph-checkpoint-sqlite' to enable persistent checkpoints. "
                f"Reason: {exception}"
            )
            yield MemorySaver()
            return

        SqliteSaver = sqlite_module.SqliteSaver

        try:
            with SqliteSaver.from_conn_string(str(checkpoint_db_path)) as checkpointer:
                logger.info(f"Using SqliteSaver for checkpointing at {checkpoint_db_path}")
                yield checkpointer
        except Exception as exception:
            logger.error(
                "Failed to initialize SQLite checkpointer. "
                f"checkpoint_db_path={checkpoint_db_path}. Reason: {exception}"
            )
            raise
