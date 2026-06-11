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
from fathom.constants.state import CommonStateKey, CompletionReason, IntentStateKey, RunOutcome
from fathom.core.config import RuntimeConfigLoader
from fathom.core.exceptions import FinalizationTimeoutError, WorkflowCancelledError
from fathom.core.services.decomposer import IntentDecomposer
from fathom.core.services.telemetry import PhaseAnnouncer
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

        self.__phase = PhaseAnnouncer(
            telemetry=telemetry,
            message=configuration.telemetry.phase,
        )

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
            phase=self.__phase,
            telemetry=telemetry,
            max_steps=max_steps,
            summarizer=summarizer,
            perception=perception,
            workflow_id=workflow_id,
            realignment=realignment,
            package_name=package_name,
            path_manager=path_manager,
            configuration=configuration,
            ocr=assembly.ocr(),
            icons=assembly.icons(),
            journal=assembly.journal(),
            ensemble=assembly.ensemble(),
            embedder=assembly.embedder(),
            pixel_overlay=assembly.overlay(),
            perception_configuration=assembly.perception_configuration,
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
        abort_warmup_task: Optional["asyncio.Task[None]"] = None

        final_state: Optional[Any] = None
        run_outcome: RunOutcome = RunOutcome.COMPLETED
        assembler = self.__ResultAssembler(workflow_id=self.__workflow_id, started_at=start_time)

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
            abort_warmup_task = asyncio.create_task(self.__graph_context.abort_detector.warmup())

            logger.info(f"[IntentStrategy] Decomposing intent: {self.__intent}")
            decomposer = IntentDecomposer.with_configuration(
                llm=self.__llm,
                configuration=self.__graph_context.configuration.llm,
            )
            await self.__phase.intent_decomposing(intent=self.__intent)
            sub_goals = await decomposer.decompose(intent=self.__intent)

            self.__graph_context.agent_state.set_sub_goals(sub_goals)
            if self.__graph_context.embedding_cache is not None:
                self.__graph_context.embedding_cache.warm(
                    texts=tuple(goal.description for goal in sub_goals),
                )
            sub_goal_payload = [
                {
                    "index": goal.index,
                    "description": goal.description,
                    "directive": goal.directive.value if goal.directive is not None else None,
                }
                for goal in sub_goals
            ]
            logger.info(
                "Intent decomposed; starting execution",
                extra={
                    "event": "intent.decomposed",
                    "component": "strategies.intent",
                    "intent": self.__intent,
                    "sub_goals": sub_goal_payload,
                    "sub_goals.count": len(sub_goals),
                    "workflow.id": self.__workflow_id,
                },
            )
            await self.__phase.plan_synthesized(
                intent=self.__intent,
                sub_goals=sub_goal_payload,
            )

            executor = GraphExecutor(
                graph=self.__graph,
                context=self.__graph_context,
                thread_id=self.__workflow_id,
                invalidate_on_injection=self.__graph_context.realignment.immediate,
                has_interrupts=self.__graph_context.signal.supports_interruption(),
            )
            run_outcome = await self.__run_executor(executor=executor)

            if run_outcome is not RunOutcome.FAILED:
                final_state = await self.__finalize_run(
                    budgets=budgets,
                    assembler=assembler,
                    run_outcome=run_outcome,
                )
        except asyncio.CancelledError:
            raise

        except Exception as exception:
            logger.exception("Intent strategy execution failed: %s", exception)
            run_outcome = RunOutcome.FAILED
            self.__try_recover_step_results()

        finally:
            await AbandonablePhase(
                workflow_id=self.__workflow_id,
                timeout=budgets.graph.checkpointer_close,
                phase=FinalizationPhase.CHECKPOINTER_CLOSE,
            ).execute(awaitable=stack.aclose())

            await AbandonablePhase(
                workflow_id=self.__workflow_id,
                phase=FinalizationPhase.BACKGROUND_DRAIN,
                timeout=budgets.runtime.background_drain,
            ).execute(
                awaitable=self.__cleanup_background_task(
                    task=prewarm_task, task_name="planner prewarm"
                )
            )

            await AbandonablePhase(
                workflow_id=self.__workflow_id,
                phase=FinalizationPhase.BACKGROUND_DRAIN,
                timeout=budgets.runtime.background_drain,
            ).execute(
                awaitable=self.__cleanup_background_task(
                    task=abort_warmup_task, task_name="abort detector warmup"
                )
            )

            await AbandonablePhase(
                workflow_id=self.__workflow_id,
                phase=FinalizationPhase.CONTEXT_SHUTDOWN,
                timeout=budgets.runtime.context_shutdown,
            ).execute(awaitable=self.__shutdown_graph_context())

        result = assembler.assemble(
            run_outcome=run_outcome,
            final_state=final_state,
            agent_state=self.__graph_context.agent_state,
        )

        self.__completion_reason = assembler.completion_reason

        if final_state is not None:
            self.__step_results = list(final_state.values.get(IntentStateKey.STEP_RESULTS) or [])

        return result

    async def __finalize_run(
        self,
        *,
        budgets: Any,
        run_outcome: RunOutcome,
        assembler: "IntentStrategy.__ResultAssembler",
    ) -> Optional[Any]:
        """
        Drain history, emit partial script + SCRIPT_GENERATED, and read final graph state.
        Runs on both successful completion and operator-driven cancellation paths.
        """

        logger.info(
            "Intent strategy finalization started",
            extra={
                "event": "fathom.intent.finalization.started",
                "workflow.id": self.__workflow_id,
                "run.outcome": run_outcome.value,
                "steps.taken": self.__graph_context.agent_state.step_count,
            },
        )

        await self.__run_finalization_phase(
            assembler=assembler,
            timeout=budgets.history.flush,
            phase=FinalizationPhase.HISTORY_FLUSH,
            awaitable=self.__graph_context.history.flush_pending_operations(),
        )

        script_data = await self.__run_finalization_phase(
            assembler=assembler,
            timeout=budgets.history.script,
            phase=FinalizationPhase.HISTORY_SCRIPT,
            awaitable=self.__graph_context.history.get_current_script(
                intent=self.__intent,
                step_number=self.__graph_context.agent_state.step_count,
            ),
        )

        await self.__emit_script_generated_event(
            script_data=script_data,
            run_outcome=run_outcome,
        )

        if self.__graph is None:
            raise RuntimeError("Intent graph is not initialized")

        config: RunnableConfig = {"configurable": {"thread_id": self.__workflow_id}}

        return await self.__run_finalization_phase(
            assembler=assembler,
            timeout=budgets.graph.state_read,
            phase=FinalizationPhase.GRAPH_STATE_READ,
            awaitable=self.__graph.aget_state(config),
        )

    async def __run_executor(self, *, executor: Any) -> RunOutcome:
        """
        Run the graph executor with start/completed/failed observability.
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

        except WorkflowCancelledError as exception:
            logger.info(
                "executor cancelled",
                extra={
                    "event": "fathom.intent.executor.cancelled",
                    "workflow.id": workflow_id,
                    "cancellation.reason": exception.reason,
                    "duration": time.perf_counter() - started_at,
                },
            )
            return RunOutcome.CANCELLED

        except Exception as exception:
            logger.error(
                "executor failed: %s",
                exception,
                exc_info=True,
                extra={
                    "event": "fathom.intent.executor.failed",
                    "workflow.id": workflow_id,
                    "exception.message": str(exception),
                    "exception.type": type(exception).__name__,
                    "duration": time.perf_counter() - started_at,
                },
            )
            return RunOutcome.FAILED

        if self.__graph_context.is_cancelled:
            logger.info(
                "executor returned with cancellation flag set",
                extra={
                    "event": "fathom.intent.executor.cancelled",
                    "workflow.id": workflow_id,
                    "duration": time.perf_counter() - started_at,
                },
            )
            return RunOutcome.CANCELLED

        logger.info(
            "executor completed",
            extra={
                "event": "fathom.intent.executor.completed",
                "workflow.id": workflow_id,
                "duration": time.perf_counter() - started_at,
            },
        )
        return RunOutcome.COMPLETED

    async def __run_finalization_phase(
        self,
        *,
        timeout: float,
        awaitable: Any,
        phase: FinalizationPhase,
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
                    "timeout": timeout,
                    "phase": phase.value,
                    "workflow.id": self.__workflow_id,
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
                    "exception.message": str(shutdown_error),
                    "exception.type": type(shutdown_error).__name__,
                },
            )

    def __try_recover_step_results(self) -> None:
        """
        Reset accumulated step results when finalization fails before they can be assembled.
        """

        self.__step_results = []

    async def __emit_script_generated_event(
        self,
        *,
        run_outcome: RunOutcome,
        script_data: Optional[str],
    ) -> None:
        """
        Emit the SCRIPT_GENERATED terminal event whenever the run reaches finalization.

        Carries an ``is_empty`` flag so consumers can distinguish "no script" from "script with content" without parsing the payload.
        ``run_outcome`` tags the event so downstream tooling can correlate empty scripts with operator-aborted runs versus successful completions.
        """

        is_empty_script = not bool(script_data and script_data.strip())

        if is_empty_script:
            logger.warning(
                "Final script generation returned empty data; emitting empty "
                "SCRIPT_GENERATED event so the client still receives a terminal signal"
            )

        logger.info(
            "Partial script artefact emitted",
            extra={
                "event": "workflow.finalization.partial_script.emitted",
                "is_empty": is_empty_script,
                "run.outcome": run_outcome.value,
                "workflow.id": self.__workflow_id,
                "steps.taken": self.__graph_context.agent_state.step_count,
            },
        )

        await self.__graph_context.telemetry.info(
            script_data or "",
            is_empty=is_empty_script,
            run_outcome=run_outcome.value,
            type=FathomEvent.SCRIPT_GENERATED,
            step=self.__graph_context.agent_state.step_count,
        )

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

        skipped: List[str] = []
        subgoals = self.__graph_context.agent_state.sub_goal_list
        executed = [
            sub_goal.description
            for sub_goal in subgoals
            if sub_goal.status == SubGoalStatus.COMPLETE
        ]

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
                serde=CheckpointSerdeFactory.build(),
                policy=checkpoint_configuration.policy,
                directory=path_manager.get_checkpoint_directory(),
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

            self.__started_at = started_at
            self.__workflow_id = workflow_id
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
            run_outcome: RunOutcome,
            final_state: Optional[Any],
        ) -> ExecutionResult:
            """
            Compose the ExecutionResult honoring partial-finalization semantics.
            """

            metadata: Dict[str, Any] = {}

            if self.__failed_phase is not None:
                metadata["finalization.partial"] = True
                metadata["finalization.failed_phase"] = self.__failed_phase

            duration = int((time.time() - self.__started_at) * 1000)
            is_cancelled = run_outcome is RunOutcome.CANCELLED

            if run_outcome is RunOutcome.FAILED:
                self.__completion_reason = CompletionReason.FAILED.value
                return ExecutionResult(
                    success=False,
                    duration=duration,
                    metadata=metadata,
                    is_cancelled=False,
                    error="executor failed before terminal state",
                )

            if run_outcome is RunOutcome.CANCELLED:
                self.__completion_reason = CompletionReason.OPERATOR_ABORTED.value
                return ExecutionResult(
                    error=None,
                    success=False,
                    duration=duration,
                    metadata=metadata,
                    is_cancelled=True,
                )

            completion_reason = self.__resolve_completion_reason(
                agent_state=agent_state,
                final_state=final_state,
            )
            self.__completion_reason = completion_reason

            success = self.__is_successful_completion(
                is_cancelled=is_cancelled,
                is_complete=agent_state.is_complete,
                completion_reason=completion_reason,
            )
            error = self.__resolve_error(
                success=success,
                final_state=final_state,
                completion_reason=completion_reason,
            )

            return ExecutionResult(
                error=error,
                metadata=metadata,
                duration=duration,
                is_cancelled=is_cancelled,
                success=success and not is_cancelled,
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
            final_state: Optional[Any],
            completion_reason: Optional[str],
        ) -> Optional[str]:
            """
            Resolve a user-facing error string from final state diagnostic and completion reason.
            """

            if success:
                return None

            if final_state is not None and (
                failure_diagnostic := final_state.values.get(
                    CommonStateKey.FAILURE_DIAGNOSTIC.value
                )
            ):
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
                CompletionReason.STUCK.value,
                CompletionReason.FAILED.value,
                CompletionReason.CANCELLED.value,
                CompletionReason.MAX_STEPS.value,
                CompletionReason.ACTION_BLOCKED.value,
                CompletionReason.USER_DIRECTIVE.value,
                CompletionReason.OPERATOR_ABORTED.value,
                CompletionReason.INTERVENTION_REQUIRED.value,
                CompletionReason.RETRY_BUDGET_EXHAUSTED.value,
            }
