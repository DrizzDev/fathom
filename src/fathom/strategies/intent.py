from __future__ import annotations

import asyncio
import time
from contextlib import AsyncExitStack
from logging import getLogger
from typing import Any, Dict, List, Optional, Tuple, cast

from langchain_core.runnables import RunnableConfig

from fathom.base.paths import SharedPathManager
from fathom.base.phase import AbandonablePhase, BoundedPhase
from fathom.constants.events import FathomEvent
from fathom.constants.finalization import FinalizationPhase
from fathom.constants.graph import NodeName
from fathom.constants.state import CommonStateKey, CompletionReason, IntentStateKey
from fathom.core.config import RuntimeConfigLoader
from fathom.core.exceptions import FinalizationTimeoutError
from fathom.core.services.decomposer import IntentDecomposer
from fathom.interfaces.checkpoint import CheckpointStore, LangGraphCheckpointer
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
from fathom.strategies.graph.context import GraphContext
from fathom.strategies.graph.intent.builder import IntentGraphBuilder

logger = getLogger(name=__name__)


# Backward-compatible re-exports for legacy callers (tests/unit/strategies/test_checkpoint_allowlist.py).
# The authoritative allow-list now lives in fathom.runtime.checkpoint_serde.CheckpointSerdeFactory.
def __load_checkpoint_allow_list() -> Tuple[Tuple[str, ...], ...]:
    """
    Resolve the checkpoint allow-list from the central serde factory.
    """

    from fathom.runtime.checkpoint_serde import CheckpointSerdeFactory

    return CheckpointSerdeFactory.allowed_json_modules()


CHECKPOINT_ALLOWED_JSON_MODULES: Tuple[Tuple[str, ...], ...] = __load_checkpoint_allow_list()
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
        runtime_configuration: Optional[RuntimeConfigLoader] = None,
        checkpoint_store: Optional[CheckpointStore] = None,
    ) -> None:
        self.__llm = llm
        self.__intent = intent
        self.__workflow_id = workflow_id

        self.__graph: Any = None
        self.__step_results: List[StepResult] = []
        self.__completion_reason: Optional[str] = None

        # Use the caller-bound loader when supplied; fall back to an
        # env-only :class:`RuntimeConfigLoader()` for stand-alone runs
        # (tests, CLI) that don't construct a settings object upstream.
        # The raw :class:`FathomSettings` is never accessible here —
        # the strategy never sees credentials material, only the
        # already-validated typed nested configs the loader returns.
        #
        # Imported lazily to break the
        # ``strategies.intent`` ↔ ``runtime.runner`` import cycle that
        # otherwise trips when callers import either package eagerly.
        from fathom.runtime.adapters import AdapterAssembly

        assembly = AdapterAssembly(
            loader=runtime_configuration
            if runtime_configuration is not None
            else RuntimeConfigLoader(),
            llm=llm,
            workflow_id=workflow_id,
            journal_directory=path_manager.base_path / "journal",
        )

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
            perception=perception,
            workflow_id=workflow_id,
            realignment=realignment,
            package_name=package_name,
            path_manager=path_manager,
            configuration=configuration,
            perception_configuration=assembly.perception_configuration,
            ocr=assembly.ocr(),
            icons=assembly.icons(),
            pixel_overlay=assembly.overlay(),
            ensemble=assembly.ensemble(),
            journal=assembly.journal(),
            artifact_pipeline=assembly.pipeline(
                path_manager=path_manager,
                storage_configuration=configuration.storage,
                artifact_configuration=configuration.artifact,
            ),
        )

        builder = IntentGraphBuilder(context=self.__graph_context)
        interrupt_nodes = [] if not signal.supports_interruption() else [NodeName.EXECUTE.value]

        # Defer checkpointer + graph construction to execute(), because the checkpointer
        # is owned by a CheckpointStore async context manager that must stay open for the duration of the graph run.
        self.__graph_builder = builder
        self.__interrupt_nodes = interrupt_nodes
        self.__checkpoint_store = checkpoint_store or self.__build_default_checkpoint_store(
            path_manager=path_manager,
            configuration=configuration,
        )

    async def execute(self) -> ExecutionResult:
        """
        Execute the intent workflow and assemble a result that honors partial-finalization semantics.
        """

        from fathom.runtime.executor import GraphExecutor

        start_time = time.time()
        prewarm_task: Optional["asyncio.Task[None]"] = None
        assembler = self.__ResultAssembler(workflow_id=self.__workflow_id, started_at=start_time)
        final_state: Optional[Any] = None
        executor_failed = False
        stack = AsyncExitStack()
        budgets = self.__graph_context.configuration.intent.finalization

        try:
            checkpointer: LangGraphCheckpointer = await stack.enter_async_context(
                self.__checkpoint_store.open(workflow_id=self.__workflow_id)
            )
            self.__graph = self.__graph_builder.build(
                checkpointer=checkpointer,
                interrupt_before=self.__interrupt_nodes,
            )

            prewarm_task = asyncio.create_task(self.__graph_context.vision.prewarm())

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
            executor_failed = await self.__run_executor(executor=executor)

            if not executor_failed:
                await self.__run_finalization_phase(
                    phase=FinalizationPhase.HISTORY_FLUSH,
                    timeout=budgets.history.flush,
                    awaitable=self.__graph_context.history.flush_pending_operations(),
                    assembler=assembler,
                )
                script_data = await self.__run_finalization_phase(
                    phase=FinalizationPhase.HISTORY_SCRIPT,
                    timeout=budgets.history.script,
                    awaitable=self.__graph_context.history.get_current_script(
                        intent=self.__intent,
                        step_number=self.__graph_context.agent_state.step_count,
                    ),
                    assembler=assembler,
                )
                if script_data:
                    await self.__graph_context.telemetry.info(
                        script_data,
                        type=FathomEvent.SCRIPT_GENERATED,
                        step=self.__graph_context.agent_state.step_count,
                    )
                elif assembler.failed_phase is None:
                    logger.warning(
                        "Final script generation returned empty data; cannot publish "
                        "SCRIPT_GENERATED event"
                    )

                if self.__graph is None:
                    raise RuntimeError("Intent graph is not initialized")

                config: RunnableConfig = {"configurable": {"thread_id": self.__workflow_id}}
                final_state = await self.__run_finalization_phase(
                    phase=FinalizationPhase.GRAPH_STATE_READ,
                    timeout=budgets.graph.state_read,
                    awaitable=self.__graph.aget_state(config),
                    assembler=assembler,
                )
        except asyncio.CancelledError:
            raise
        except Exception as exception:
            logger.exception("Intent strategy execution failed: %s", exception)
            executor_failed = True
            self.__try_recover_step_results()
        finally:
            await AbandonablePhase(
                phase=FinalizationPhase.CHECKPOINTER_CLOSE,
                timeout=budgets.graph.checkpointer_close,
                workflow_id=self.__workflow_id,
            ).execute(awaitable=stack.aclose())

            await AbandonablePhase(
                phase=FinalizationPhase.BACKGROUND_DRAIN,
                timeout=budgets.runtime.background_drain,
                workflow_id=self.__workflow_id,
            ).execute(
                awaitable=self.__cleanup_background_task(
                    task=prewarm_task, task_name="planner prewarm"
                )
            )

            await AbandonablePhase(
                phase=FinalizationPhase.CONTEXT_SHUTDOWN,
                timeout=budgets.runtime.context_shutdown,
                workflow_id=self.__workflow_id,
            ).execute(awaitable=self.__shutdown_graph_context())

        result = assembler.assemble(
            agent_state=self.__graph_context.agent_state,
            is_cancelled=self.__graph_context.is_cancelled,
            final_state=final_state,
            executor_failed=executor_failed,
        )
        self.__completion_reason = assembler.completion_reason
        if final_state is not None:
            self.__step_results = list(final_state.values.get(IntentStateKey.STEP_RESULTS) or [])
        return result

    async def __run_executor(self, *, executor: Any) -> bool:
        """
        Run the graph executor with start/completed/failed observability; return True on failure.
        """

        workflow_id = self.__workflow_id
        logger.info(
            "executor started",
            extra={"event": "fathom.intent.executor.started", "workflow.id": workflow_id},
        )
        started_at = time.perf_counter()
        try:
            await executor.run()
        except asyncio.CancelledError:
            raise
        except Exception as exception:
            logger.error(
                "executor failed: %s",
                exception,
                exc_info=True,
                extra={
                    "event": "fathom.intent.executor.failed",
                    "workflow.id": workflow_id,
                    "duration": time.perf_counter() - started_at,
                    "exception.type": type(exception).__name__,
                    "exception.message": str(exception),
                },
            )
            return True
        logger.info(
            "executor completed",
            extra={
                "event": "fathom.intent.executor.completed",
                "workflow.id": workflow_id,
                "duration": time.perf_counter() - started_at,
            },
        )
        return False

    async def __run_finalization_phase(
        self,
        *,
        phase: FinalizationPhase,
        timeout: float,
        awaitable: Any,
        assembler: "IntentStrategy.__ResultAssembler",
    ) -> Any:
        """
        Wrap one finalization await with BoundedPhase; record phase identity on timeout but never raise.
        """

        try:
            return await BoundedPhase(
                phase=phase,
                timeout=timeout,
                workflow_id=self.__workflow_id,
            ).execute(awaitable=awaitable)
        except FinalizationTimeoutError as exception:
            assembler.record_finalization_failure(phase=exception.phase)
            logger.warning(
                "finalization phase timed out",
                extra={
                    "event": f"{phase.value}.partial",
                    "phase": phase.value,
                    "workflow.id": self.__workflow_id,
                    "timeout": timeout,
                },
            )
            return None

    async def __shutdown_graph_context(self) -> None:
        """
        Wrapper around graph context shutdown so it can be passed to AbandonablePhase as an awaitable.
        """

        try:
            await self.__graph_context.shutdown()
        except Exception as shutdown_error:
            logger.warning(
                "graph context shutdown failed: %s",
                shutdown_error,
                extra={
                    "event": f"{FinalizationPhase.CONTEXT_SHUTDOWN.value}.failed",
                    "workflow.id": self.__workflow_id,
                    "exception.type": type(shutdown_error).__name__,
                    "exception.message": str(shutdown_error),
                },
            )

    def __try_recover_step_results(self) -> None:
        """
        Reset accumulated step results when finalization fails before they can be assembled.
        """

        self.__step_results = []

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
    def __build_default_checkpoint_store(
        *,
        path_manager: SharedPathManager,
        configuration: FathomConfiguration,
    ) -> CheckpointStore:
        """
        Construct the default CheckpointStore from configuration when callers do not inject one.
        """

        from fathom.adapters.checkpoint import SqliteCheckpointStore
        from fathom.runtime.checkpoint_serde import CheckpointSerdeFactory
        from fathom.schemas.checkpoint import SqliteCheckpointConfiguration

        checkpoint_configuration = configuration.intent.checkpoint
        if isinstance(checkpoint_configuration, SqliteCheckpointConfiguration):
            return SqliteCheckpointStore(
                directory=path_manager.get_checkpoint_directory(),
                policy=checkpoint_configuration.policy,
                serde=CheckpointSerdeFactory.build(),
            )

        raise RuntimeError(
            f"No default CheckpointStore implementation for backend "
            f"'{getattr(checkpoint_configuration, 'backend', 'unknown')}'. "
            "Inject a CheckpointStore explicitly."
        )

    class __ResultAssembler:
        """
        Build an ExecutionResult from strategy state plus any partial-finalization markers.
        """

        def __init__(self, *, workflow_id: str, started_at: float) -> None:
            """
            Bind workflow identity and execution start timestamp for duration accounting.
            """

            self.__workflow_id = workflow_id
            self.__started_at = started_at
            self.__failed_phase: Optional[str] = None
            self.__completion_reason: Optional[str] = None

        @property
        def failed_phase(self) -> Optional[str]:
            """
            Identifier of the first finalization phase that did not complete cleanly, or None.
            """

            return self.__failed_phase

        @property
        def completion_reason(self) -> Optional[str]:
            """
            Completion reason captured during result assembly.
            """

            return self.__completion_reason

        def record_finalization_failure(self, *, phase: str) -> None:
            """
            Mark that one finalization phase did not complete cleanly.
            """

            if self.__failed_phase is None:
                self.__failed_phase = phase

        def assemble(
            self,
            *,
            agent_state: Any,
            is_cancelled: bool,
            final_state: Optional[Any],
            executor_failed: bool,
        ) -> ExecutionResult:
            """
            Compose the ExecutionResult honoring partial-finalization semantics.
            """

            metadata: Dict[str, Any] = {}
            if self.__failed_phase is not None:
                metadata["finalization.partial"] = True
                metadata["finalization.failed_phase"] = self.__failed_phase
            duration = int((time.time() - self.__started_at) * 1000)

            if executor_failed:
                self.__completion_reason = CompletionReason.FAILED.value
                return ExecutionResult(
                    duration=duration,
                    is_cancelled=is_cancelled,
                    success=False,
                    error="executor failed before terminal state",
                    metadata=metadata,
                )

            completion_reason = self.__resolve_completion_reason(
                agent_state=agent_state,
                final_state=final_state,
            )
            self.__completion_reason = completion_reason
            success = self.__is_successful_completion(
                is_complete=agent_state.is_complete,
                is_cancelled=is_cancelled,
                completion_reason=completion_reason,
            )
            error = self.__resolve_error(
                success=success,
                completion_reason=completion_reason,
                final_state=final_state,
            )

            return ExecutionResult(
                duration=duration,
                is_cancelled=is_cancelled,
                success=success and not is_cancelled,
                error=error,
                metadata=metadata,
            )

        @staticmethod
        def __resolve_completion_reason(
            *,
            agent_state: Any,
            final_state: Optional[Any],
        ) -> Optional[str]:
            """
            Prefer the checkpoint completion reason; fall back to in-memory agent state when state read was skipped.
            """

            if final_state is not None:
                reason = final_state.values.get("completion_reason")
                if reason is not None:
                    return cast("Optional[str]", reason)
            return cast("Optional[str]", agent_state.completion_reason)

        @staticmethod
        def __resolve_error(
            *,
            success: bool,
            completion_reason: Optional[str],
            final_state: Optional[Any],
        ) -> Optional[str]:
            """
            Resolve a user-facing error string from final state diagnostic and completion reason.
            """

            if success:
                return None
            if final_state is not None:
                failure_diagnostic = final_state.values.get(CommonStateKey.FAILURE_DIAGNOSTIC.value)
                if failure_diagnostic:
                    return str(failure_diagnostic)
            return completion_reason

        @staticmethod
        def __is_successful_completion(
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
                CompletionReason.USER_DIRECTIVE.value,
            }
