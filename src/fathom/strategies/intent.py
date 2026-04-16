from __future__ import annotations

import asyncio
import importlib
import inspect
import time
from contextlib import AsyncExitStack, asynccontextmanager
from logging import getLogger
from pathlib import Path  # noqa: TC003
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver

from fathom.base.paths import SharedPathManager
from fathom.constants.events import FathomEvent
from fathom.constants.graph import NodeName
from fathom.constants.state import CompletionReason, IntentStateKey
from fathom.core.services.decomposer import IntentDecomposer
from fathom.core.services.intent_classifier import IntentClassifier
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
from fathom.schemas.results import ExecutionResult
from fathom.schemas.run import RealignmentPolicy
from fathom.schemas.steps import StepResult
from fathom.schemas.subgoal import SubGoal, SubGoalStatus
from fathom.settings.env import FathomSettings
from fathom.strategies.graph.context import GraphContext
from fathom.strategies.graph.intent.builder import IntentGraphBuilder

logger = getLogger(name=__name__)

CHECKPOINT_ALLOWED_JSON_MODULES: Tuple[Tuple[str, ...], ...] = (
    ("fathom.schemas.screens", "ScreenCapture"),
    ("fathom.schemas.screens", "ScreenState"),
    ("fathom.schemas.results", "PlanResult"),
    ("fathom.schemas.steps", "Step"),
    ("fathom.schemas.steps", "StepResult"),
    ("fathom.constants", "ActionType"),
    ("fathom.constants.state", "CommonStateKey"),
    ("fathom.constants.state", "IntentStateKey"),
)
CHECKPOINT_ALLOWED_MSGPACK_MODULES: Tuple[Tuple[str, ...], ...] = CHECKPOINT_ALLOWED_JSON_MODULES


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
        interrupt_nodes = [] if not signal.supports_interruption() else [NodeName.EXECUTE.value]

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
        prewarm_task: Optional["asyncio.Task[None]"] = None

        try:
            async with AsyncExitStack() as stack:
                checkpointer: Any = await stack.enter_async_context(
                    self.__build_checkpointer_context(checkpoint_db_path=self.__checkpoint_db)
                )
                self.__graph = self.__graph_builder.build(
                    checkpointer=checkpointer,
                    interrupt_before=self.__interrupt_nodes,
                )

                # Prewarm prompt cache concurrently with decomposition to reduce first-call latency.
                prewarm_task = asyncio.create_task(self.__graph_context.vision.prewarm())

                sub_goals = await self.__resolve_initial_sub_goals()
                self.__graph_context.agent_state.set_sub_goals(sub_goals)
                logger.info(
                    f"[IntentStrategy] Intent resolved to {len(sub_goals)} sub-goals. "
                    "Starting execution..."
                )

                # Cue: first sub-goal becomes active. Subsequent
                # transitions fire from the VERIFY node when a sub-goal
                # is marked complete.
                if sub_goals:
                    first_goal = sub_goals[0]
                    await self.__graph_context.telemetry.info(
                        "Sub-goal started",
                        type=FathomEvent.SUB_GOAL_STARTED,
                        index=first_goal.index,
                        total=len(sub_goals),
                        description=first_goal.description,
                    )

                executor = GraphExecutor(
                    graph=self.__graph,
                    context=self.__graph_context,
                    thread_id=self.__workflow_id,
                    invalidate_on_injection=self.__graph_context.realignment.immediate,
                    has_interrupts=self.__graph_context.signal.supports_interruption(),
                )
                await executor.run()
                await self.__graph_context.history.flush_pending_operations()

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

                config: RunnableConfig = {"configurable": {"thread_id": self.__workflow_id}}
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
                await self.__graph_context.history.flush_pending_operations()
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
        finally:
            await self.__cleanup_background_task(task=prewarm_task, task_name="planner prewarm")

            try:
                await self.__graph_context.shutdown()
            except Exception as shutdown_error:
                logger.warning(f"[intent-strategy] graph context shutdown failed: {shutdown_error}")

    async def __cleanup_background_task(
        self,
        *,
        task_name: str,
        task: Optional["asyncio.Task[Any]"],
    ) -> None:
        """
        Finish or cancel a background task used by the intent strategy.
        """

        if task is None:
            return

        if task.done():
            try:
                await task
            except Exception as exception:
                logger.warning("Background %s task finished with error: %s", task_name, exception)
            return

        task.cancel()

        try:
            await task
        except asyncio.CancelledError:
            logger.warning("Background %s task cancelled", task_name)
        except Exception as exception:
            logger.warning("Background %s task failed during cleanup: %s", task_name, exception)

    async def __resolve_initial_sub_goals(self) -> List[SubGoal]:
        """Classify the intent, then either wrap it as a single sub-goal
        or delegate to ``IntentDecomposer`` to break it into steps.

        Gated by ``FathomSettings.allow_atomic_intent_single_subgoal``.
        When the flag is off, always decomposes. When on, first calls
        ``IntentClassifier.should_decompose`` — a tool-call-based LLM
        check — and wraps the intent as a single
        ``SubGoal`` if the classifier says it's simple enough for the
        planner to execute end-to-end. Fails safe via the classifier's
        own fail-safe semantics (any classifier failure returns True,
        so we fall through to the decomposer).
        """

        settings = FathomSettings()
        if settings.allow_atomic_intent_single_subgoal:
            logger.info(f"[IntentStrategy] Classifying intent: {self.__intent}")
            classifier = IntentClassifier(llm=self.__llm)
            should_decompose = await classifier.should_decompose(intent=self.__intent)
        else:
            should_decompose = True

        # Cue: surface the classifier decision regardless of log level
        # so the demo audience sees what the agent just decided.
        await self.__graph_context.telemetry.info(
            "Classifier decision",
            type=FathomEvent.INTENT_CLASSIFIED,
            should_decompose=should_decompose,
            intent=self.__intent,
        )

        if not should_decompose:
            logger.info(
                "[IntentStrategy] Classifier marked intent as simple; skipping decomposition."
            )
            sub_goals = [
                SubGoal(
                    index=0,
                    description=self.__intent,
                    status=SubGoalStatus.PENDING,
                    confidence=0.9,
                )
            ]
            await self.__graph_context.telemetry.info(
                "Plan ready",
                type=FathomEvent.DECOMPOSITION_COMPLETE,
                sub_goals=[goal.description for goal in sub_goals],
                decomposed=False,
            )
            return sub_goals

        logger.info(f"[IntentStrategy] Decomposing intent: {self.__intent}")
        decomposer = IntentDecomposer.with_configuration(
            llm=self.__llm,
            configuration=self.__graph_context.configuration.llm,
        )
        sub_goals = await decomposer.decompose(intent=self.__intent)
        await self.__graph_context.telemetry.info(
            "Plan ready",
            type=FathomEvent.DECOMPOSITION_COMPLETE,
            sub_goals=[goal.description for goal in sub_goals],
            decomposed=True,
        )
        return sub_goals

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

    def get_subgoal_execution_audit(self) -> Tuple[List[str], List[str], int]:
        """
        Get audit trail of executed vs skipped subgoals.
        """

        from fathom.schemas.subgoal import SubGoalStatus

        subgoals = self.__graph_context.agent_state.sub_goal_list
        executed = [sg.description for sg in subgoals if sg.status == SubGoalStatus.COMPLETE]
        skipped: List[str] = []
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

    @staticmethod
    def __build_checkpoint_serde() -> Any:
        """
        Build a JsonPlusSerializer that whitelists Fathom types for checkpoint
        deserialization, suppressing 'Deserializing unregistered type' warnings.
        Uses the module-level CHECKPOINT_ALLOWED_* constants for consistency.
        """

        from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

        serializer_configuration: Dict[str, Any] = {
            "allowed_json_modules": CHECKPOINT_ALLOWED_JSON_MODULES,
        }

        serializer_signature = inspect.signature(JsonPlusSerializer)
        if "allowed_msgpack_modules" in serializer_signature.parameters:
            serializer_configuration["allowed_msgpack_modules"] = CHECKPOINT_ALLOWED_MSGPACK_MODULES

        return JsonPlusSerializer(**serializer_configuration)

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
            sqlite_module = importlib.import_module("langgraph.checkpoint.sqlite.aio")
            aiosqlite_module = importlib.import_module("aiosqlite")
        except (ImportError, ModuleNotFoundError) as exception:
            logger.warning(
                "AsyncSqliteSaver unavailable; falling back to MemorySaver. "
                "Install 'langgraph-checkpoint-sqlite' and 'aiosqlite' to enable "
                f"persistent checkpoints. Reason: {exception}"
            )
            yield MemorySaver(serde=self.__build_checkpoint_serde())
            return

        AsyncSqliteSaver = sqlite_module.AsyncSqliteSaver
        serde = self.__build_checkpoint_serde()

        try:
            async with aiosqlite_module.connect(str(checkpoint_db_path)) as connection:
                checkpointer = AsyncSqliteSaver(connection, serde=serde)
                logger.info(
                    "Using AsyncSqliteSaver for checkpointing at %s",
                    checkpoint_db_path,
                )
                yield checkpointer
        except Exception as exception:
            logger.error(
                "Failed to initialize AsyncSqliteSaver. "
                f"checkpoint_db_path={checkpoint_db_path}. Reason: {exception}"
            )
            raise
