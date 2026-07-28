from __future__ import annotations

import asyncio
import time
from contextlib import AsyncExitStack
from logging import getLogger
from typing import Any, Awaitable, Dict, List, Optional, Tuple, TypeVar, cast

import structlog
from langchain_core.runnables import RunnableConfig

from fathom.authoring.application import StepDraftComposer
from fathom.base.paths import SharedPathManager
from fathom.base.phase import AbandonablePhase, BoundedPhase
from fathom.constants.authoring import AuthoringKind, AuthoringStatus
from fathom.constants.events import FathomEvent
from fathom.constants.flow import IssueCode
from fathom.constants.generation import ScriptSource, ScriptStatus
from fathom.constants.graph import NodeName
from fathom.constants.state import (
    CommonStateKey,
    CompletionReason,
    IntentStateKey,
    RunOutcome,
)
from fathom.constants.turn.termination import TerminationStatus
from fathom.core.agent.termination import TerminationResolver
from fathom.core.capability.catalog import CommandCatalog
from fathom.core.config import RuntimeConfigLoader
from fathom.core.exceptions import (
    FinalizationTimeoutError,
    LanguageComplianceError,
    WorkflowCancelledError,
)
from fathom.core.services.decomposer import IntentDecomposer
from fathom.core.services.directive import DirectivePolicy
from fathom.core.services.recorder import ConversationRecorder
from fathom.core.services.telemetry import PhaseAnnouncer
from fathom.interfaces.authoring import AuthoringPort
from fathom.interfaces.checkpoint import CheckpointStore, LangGraphCheckpointer
from fathom.interfaces.device import DevicePort
from fathom.interfaces.llm import LLMPort
from fathom.interfaces.memory import MemoryPort
from fathom.interfaces.perception import PerceptionPort
from fathom.interfaces.signal import SignalPort
from fathom.interfaces.storage import StoragePort
from fathom.interfaces.summarization import SummarizationPort
from fathom.interfaces.telemetry import TelemetryPort
from fathom.runtime.checkpoint_serde import CheckpointSerdeFactory
from fathom.schemas.authoring import AuthoringBaseline, AuthoringBaselineCommand, AuthoringTask
from fathom.schemas.configuration import FathomConfiguration
from fathom.schemas.finalization import FinalizationBudgetPolicy
from fathom.schemas.flow import Check, CheckNode, Evidence, Flow, Issue, RunObjective
from fathom.schemas.generation import (
    CompletionValidation,
    GenerationResult,
    ScriptFileMetadata,
    ScriptReview,
)
from fathom.schemas.metrics import ExecutionMetrics
from fathom.schemas.results import ExecutionResult
from fathom.schemas.run import RealignmentPolicy
from fathom.schemas.steps import StepResult
from fathom.strategies.graph.context import GraphContext
from fathom.strategies.graph.intent.builder import IntentGraphBuilder

logger = getLogger(name=__name__)

_FinalizationResult = TypeVar("_FinalizationResult")
_CANCELLED_SCRIPT_HEARTBEAT_FAILURE_LIMIT = 3


CHECKPOINT_ALLOWED_JSON_MODULES: Tuple[Tuple[str, ...], ...] = (
    CheckpointSerdeFactory.allowed_json_modules()
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
        tenant: str,
        thread: str,
        catalog: CommandCatalog,
        use_xml: bool,
        max_steps: int,
        requester: str,
        responder: str,
        workflow_id: str,
        execution_id: str,
        package_name: str,
        workspace: Optional[str] = None,
        recorder: Optional[ConversationRecorder] = None,
        realignment: Optional[RealignmentPolicy] = None,
        authoring: Optional[AuthoringPort] = None,
        checkpoint_store: Optional[CheckpointStore] = None,
        runtime_configuration: Optional[RuntimeConfigLoader] = None,
    ) -> None:
        self.__llm = llm
        self.__intent = intent
        self.__catalog = catalog
        self.__workflow_id = workflow_id
        self.__execution_id = execution_id

        self.__graph: Any = None
        self.__step_results: List[StepResult] = []

        self.__final_script: Optional[str] = None
        self.__completion_reason: Optional[str] = None
        self.__draft_composer = StepDraftComposer()

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
            tenant=tenant,
            thread=thread,
            intent=intent,
            device=device,
            memory=memory,
            signal=signal,
            use_xml=use_xml,
            storage=storage,
            recorder=recorder,
            phase=self.__phase,
            telemetry=telemetry,
            max_steps=max_steps,
            requester=requester,
            responder=responder,
            workspace=workspace,
            summarizer=summarizer,
            perception=perception,
            workflow_id=workflow_id,
            realignment=realignment,
            execution_id=execution_id,
            package_name=package_name,
            path_manager=path_manager,
            configuration=configuration,
            ocr=assembly.ocr(),
            icons=assembly.icons(),
            journal=assembly.journal(),
            ensemble=assembly.ensemble(),
            embedder=assembly.embedder(),
            pixel_overlay=assembly.overlay(),
            authoring=authoring,
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

        # Bind the run identity so every log this run emits — including the stdlib
        # LoopDetector logs on the shared device stream — carries workflow.id.
        structlog.contextvars.bind_contextvars(**{"workflow.id": self.__workflow_id})

        try:
            checkpointer: LangGraphCheckpointer = await stack.enter_async_context(
                self.__checkpoint_store.open(workflow_id=self.__workflow_id)
            )
            self.__graph = self.__graph_builder.build(
                checkpointer=cast("Any", checkpointer),
                interrupt_before=self.__interrupt_nodes,
            )

            prewarm_task = asyncio.create_task(self.__graph_context.vision.prewarm())
            abort_warmup_task = asyncio.create_task(self.__graph_context.abort_detector.warmup())

            logger.info(
                f"decomposing intent: {self.__intent}",
                extra={
                    "event": "intent.decompose.started",
                    "workflow.id": self.__workflow_id,
                },
            )
            decomposer = IntentDecomposer.with_configuration(
                llm=self.__llm,
                directive_policy=DirectivePolicy(catalog=self.__catalog),
                configuration=self.__graph_context.configuration.llm,
            )
            await self.__phase.intent_decomposing(intent=self.__intent)
            sub_goals = await decomposer.decompose(intent=self.__intent)

            self.__graph_context.agent_state.set_sub_goals(sub_goals)

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
                    "workflow.id": self.__workflow_id,
                    "intent": self.__intent,
                    "sub_goals": sub_goal_payload,
                    "sub_goals.count": len(sub_goals),
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
            final_state = await self.__finalize_run(
                budgets=budgets,
                assembler=assembler,
                run_outcome=run_outcome,
            )
        except asyncio.CancelledError:
            self.__graph_context.cancel()
            final_state = await self.__finalize_cancelled_run(
                budgets=budgets,
                assembler=assembler,
            )
            raise

        except Exception as exception:
            logger.exception(
                "intent strategy execution failed",
                extra={
                    "event": "intent.execution.failed",
                    "workflow.id": self.__workflow_id,
                    "exception.type": type(exception).__name__,
                    "exception.message": str(exception),
                },
            )
            run_outcome = RunOutcome.FAILED
            final_state = await self.__finalize_failed_run(
                budgets=budgets,
                assembler=assembler,
            )

        finally:
            structlog.contextvars.unbind_contextvars("workflow.id")

            await AbandonablePhase(
                workflow_id=self.__workflow_id,
                timeout=budgets.graph.checkpointer_close,
                phase="fathom.finalization.checkpointer.close",
            ).execute(awaitable=stack.aclose())

            await AbandonablePhase(
                workflow_id=self.__workflow_id,
                phase="fathom.runner.background.drain",
                timeout=budgets.runtime.background_drain,
            ).execute(
                awaitable=self.__cleanup_background_task(
                    task=prewarm_task, task_name="planner prewarm"
                )
            )

            await AbandonablePhase(
                workflow_id=self.__workflow_id,
                phase="fathom.runner.background.drain",
                timeout=budgets.runtime.background_drain,
            ).execute(
                awaitable=self.__cleanup_background_task(
                    task=abort_warmup_task, task_name="abort detector warmup"
                )
            )

            await AbandonablePhase(
                workflow_id=self.__workflow_id,
                phase="fathom.runner.context.shutdown",
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
        run_outcome: RunOutcome,
        budgets: FinalizationBudgetPolicy,
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
            phase="fathom.finalization.history.flush",
            awaitable=self.__graph_context.history.flush_pending_operations(),
        )

        final_state = await self.__read_final_graph_state(
            assembler=assembler,
            timeout=budgets.graph.state_read,
        )
        script_outcome = self.__resolve_script_outcome(
            run_outcome=run_outcome,
            final_state=final_state,
        )

        if run_outcome is RunOutcome.CANCELLED:
            await self.__publish_cancelled_run_script(
                budgets=budgets,
                assembler=assembler,
            )
        else:
            script_data = await self.__run_finalization_phase(
                assembler=assembler,
                timeout=budgets.history.script,
                awaitable=self.__author_script(run_outcome=script_outcome),
                phase="fathom.finalization.history.script",
            )

            await self.__deliver_final_script(
                quality=script_data,
                run_outcome=script_outcome,
            )

        return final_state

    async def __finalize_failed_run(
        self,
        *,
        budgets: FinalizationBudgetPolicy,
        assembler: "IntentStrategy.__ResultAssembler",
    ) -> Optional[Any]:
        """
        Best-effort script finalization for failures raised outside the graph terminal path.
        """

        try:
            return await self.__finalize_run(
                budgets=budgets,
                assembler=assembler,
                run_outcome=RunOutcome.FAILED,
            )
        except Exception as exception:
            logger.exception(
                "failed-run script finalization failed",
                extra={
                    "event": "fathom.intent.failed_finalization.failed",
                    "workflow.id": self.__workflow_id,
                    "exception.message": str(exception),
                    "exception.type": type(exception).__name__,
                },
            )
            self.__try_recover_step_results()
            return None

    async def __finalize_cancelled_run(
        self,
        *,
        budgets: FinalizationBudgetPolicy,
        assembler: "IntentStrategy.__ResultAssembler",
    ) -> Optional[Any]:
        """
        Best-effort finalization for host-level cancellation before re-raising cancellation.
        """

        if self.__graph is None:
            logger.warning(
                "Skipping cancelled-run finalization because graph was not initialized",
                extra={
                    "event": "fathom.intent.cancelled_finalization.skipped",
                    "workflow.id": self.__workflow_id,
                    "reason": "graph_not_initialized",
                },
            )
            return None

        try:
            return await self.__finalize_run(
                budgets=budgets,
                assembler=assembler,
                run_outcome=RunOutcome.CANCELLED,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exception:
            logger.exception(
                "Cancelled-run finalization failed",
                extra={
                    "event": "fathom.intent.cancelled_finalization.failed",
                    "workflow.id": self.__workflow_id,
                    "exception.message": str(exception),
                    "exception.type": type(exception).__name__,
                },
            )
            return None

    async def __read_final_graph_state(
        self,
        *,
        timeout: float,
        assembler: "IntentStrategy.__ResultAssembler",
    ) -> Optional[Any]:
        """
        Read the terminal graph state when available without blocking script finalization.
        """

        if self.__graph is None:
            logger.warning(
                "final graph state unavailable because graph was not initialized",
                extra={
                    "event": "fathom.finalization.graph.state.unavailable",
                    "workflow.id": self.__workflow_id,
                    "reason": "graph_not_initialized",
                },
            )
            return None

        phase = "fathom.finalization.graph.state"
        config: RunnableConfig = {"configurable": {"thread_id": self.__workflow_id}}

        try:
            return await self.__run_finalization_phase(
                assembler=assembler,
                timeout=timeout,
                phase=phase,
                awaitable=self.__graph.aget_state(config),
            )
        except Exception as exception:
            assembler.record_finalization_failure(phase=phase)
            logger.warning(
                "final graph state read failed; continuing script finalization",
                extra={
                    "event": "fathom.finalization.graph.state.unavailable",
                    "workflow.id": self.__workflow_id,
                    "exception.type": type(exception).__name__,
                    "exception.message": str(exception),
                },
            )
            return None

    def __resolve_script_outcome(
        self, *, run_outcome: RunOutcome, final_state: Optional[Any]
    ) -> RunOutcome:
        """
        Derive the script event outcome from the honest termination status, resolved in one place.
        """

        completion_reason = (
            (
                final_state.values.get(CommonStateKey.COMPLETION_REASON)
                or final_state.values.get(CommonStateKey.COMPLETION_REASON.value)
                or final_state.values.get("completion_reason")
            )
            if final_state is not None
            else None
        )
        status = TerminationResolver().resolve(outcome=run_outcome, reason=completion_reason)

        logger.info(
            "Termination resolved",
            extra={
                "event": "fathom.intent.termination.resolved",
                "workflow.id": self.__workflow_id,
                "termination.status": status.value,
                "termination.reason": completion_reason,
                "run.outcome": run_outcome.value,
            },
        )

        if run_outcome is RunOutcome.COMPLETED and status is not TerminationStatus.COMPLETED:
            return RunOutcome.FAILED

        return run_outcome

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
        phase: str,
        awaitable: Awaitable[_FinalizationResult],
        assembler: "IntentStrategy.__ResultAssembler",
    ) -> Optional[_FinalizationResult]:
        """
        Wrap one finalization await with BoundedPhase; record phase identity on timeout but never raise.
        """

        started = time.perf_counter()
        logger.info(
            "finalization phase started",
            extra={
                "event": "script.finalization.phase.started",
                "timeout": timeout,
                "phase": phase,
                "workflow.id": self.__workflow_id,
            },
        )

        try:
            result = await self.__await_finalization_phase(
                phase=phase,
                timeout=timeout,
                awaitable=awaitable,
            )
        except FinalizationTimeoutError as exception:
            assembler.record_finalization_failure(phase=exception.phase)
            logger.warning(
                "finalization phase timed out",
                extra={
                    "event": "script.finalization.phase.timed_out",
                    "timeout": timeout,
                    "phase": phase,
                    "workflow.id": self.__workflow_id,
                    "duration.ms": round((time.perf_counter() - started) * 1000, 3),
                },
            )
            return None
        except Exception as exception:
            logger.warning(
                "finalization phase failed",
                extra={
                    "event": "script.finalization.phase.failed",
                    "phase": phase,
                    "workflow.id": self.__workflow_id,
                    "exception.type": type(exception).__name__,
                    "exception.message": str(exception),
                    "duration.ms": round((time.perf_counter() - started) * 1000, 3),
                },
            )
            raise

        logger.info(
            "finalization phase completed",
            extra={
                "event": "script.finalization.phase.completed",
                "phase": phase,
                "duration.ms": round((time.perf_counter() - started) * 1000, 3),
                "workflow.id": self.__workflow_id,
            },
        )
        return result

    async def __publish_cancelled_run_script(
        self,
        *,
        budgets: FinalizationBudgetPolicy,
        assembler: "IntentStrategy.__ResultAssembler",
    ) -> None:
        """
        Publish a cancelled run's script by generating quality first, then falling back to baseline.

        The cancelled-run failure catches wrap only the script generation step.
        Flush failures must surface; they are not absorbed into empty-script delivery.
        """

        script_task = asyncio.create_task(
            self.__await_finalization_phase(
                timeout=budgets.history.script,
                awaitable=self.__author_script(run_outcome=RunOutcome.CANCELLED),
                phase="fathom.finalization.history.script",
            ),
            name="fathom.finalization.history.script",
        )
        heartbeat_task = asyncio.create_task(
            self.__emit_cancelled_script_heartbeats(script_task=script_task),
            name="fathom.finalization.history.script.heartbeat",
        )

        started_at = time.perf_counter()
        try:
            quality = await self.__cancelled_quality_script(
                assembler=assembler,
                started_at=started_at,
                script_task=script_task,
                timeout=budgets.history.script,
            )
        finally:
            await self.__stop_cancelled_script_heartbeat(heartbeat_task=heartbeat_task)

        await self.__deliver_cancelled_script(
            quality=quality,
            assembler=assembler,
            started_at=started_at,
        )

    async def __author_script(self, *, run_outcome: RunOutcome) -> Optional[GenerationResult]:
        """
        Ask the authoring runner for a reviewed script, returning None when fallback should run.
        """

        evidence = self.__evidence_for_outcome(
            run_outcome=run_outcome,
            evidence=await self.__graph_context.evidence.read(
                execution_id=self.__execution_id,
                objective=RunObjective(
                    goal=self.__intent,
                    intent=self.__intent,
                    package=self.__graph_context.package_name,
                ),
            ),
        )
        review = ScriptReview(
            partial=evidence.partial,
            reason=evidence.reason,
            discarded=evidence.discarded,
        )
        drafts = await self.__graph_context.authoring_drafts.list(execution_id=self.__execution_id)
        baseline = await self.__baseline_for_authoring()
        response = await self.__graph_context.authoring_runner.author(
            author=self.__graph_context.authoring,
            task=AuthoringTask(
                intent=self.__intent,
                kind=AuthoringKind.RUN,
                execution_id=self.__execution_id,
                step_number=self.__graph_context.agent_state.step_count,
                evidence=self.__graph_context.authoring_evidence_builder.build_run(
                    drafts=drafts,
                    evidence=evidence,
                    baseline=baseline,
                ),
            ),
        )
        if response.status is AuthoringStatus.GENERATED and response.has_script:
            artifact = response.artifact
            if artifact is not None:
                review = review.model_copy(
                    update={
                        "lineage": artifact.lineage,
                        "commands": artifact.commands,
                        "advisories": artifact.advisories,
                    }
                )

            return GenerationResult(
                attempts=1,
                review=review,
                text=response.script or "",
                source=ScriptSource.QUALITY,
            )

        logger.info(
            "final authoring did not produce a script",
            extra={
                "event": "authoring.run.unavailable",
                "execution.id": self.__execution_id,
                "authoring.reason": response.reason,
                "authoring.status": response.status.value,
            },
        )
        draft_result = self.__draft_composer.compose(
            drafts=drafts,
            evidence=evidence,
            baseline=baseline,
            completion=CompletionValidation(
                lines=self.__terminal_assertion_lines(evidence=evidence),
                required=self.__requires_terminal_assertion(evidence=evidence),
                source_steps=self.__assertion_source_steps(evidence=evidence),
            ),
        )
        if draft_result is not None and self.__valid_script(result=draft_result):
            logger.info(
                "finalization selected composed step drafts after run authoring failed",
                extra={
                    "event": "authoring.step_drafts.selected",
                    "execution.id": self.__execution_id,
                    "script.partial": draft_result.review.partial,
                    "script.line_count": len(draft_result.text.splitlines()),
                },
            )
            return draft_result

        return None

    def __terminal_assertion_lines(self, *, evidence: Evidence) -> Tuple[str, ...]:
        """
        Render verifier completion assertions as terminal Drizz validation lines.
        """

        if not self.__requires_terminal_assertion(evidence=evidence):
            return ()

        node = CheckNode(
            source_steps=self.__assertion_source_steps(evidence=evidence),
            assertion_ids=tuple(assertion.id for assertion in evidence.assertions),
            checks=tuple(
                Check(kind=assertion.kind, subject=assertion.subject)
                for assertion in evidence.assertions
            ),
        )
        flow = Flow(
            nodes=(node,),
            intent=self.__intent,
            package=self.__graph_context.package_name,
        )

        try:
            text = self.__graph_context.dialect.renderer.render(flow=flow)
        except LanguageComplianceError as exception:
            logger.exception(
                "Terminal completion assertions could not be rendered for step drafts",
                extra={
                    "event": "authoring.step_drafts.terminal_unavailable",
                    "reason": str(exception),
                    "execution.id": self.__execution_id,
                },
            )
            return ()

        return tuple(line.strip() for line in text.splitlines() if line.strip())

    @staticmethod
    def __requires_terminal_assertion(*, evidence: Evidence) -> bool:
        """
        Return whether composed drafts need a terminal validation command.
        """

        return evidence.outcome is RunOutcome.COMPLETED and bool(evidence.assertions)

    @staticmethod
    def __assertion_source_steps(*, evidence: Evidence) -> Tuple[int, ...]:
        """
        Return valid source steps for verifier completion assertions.
        """

        steps: List[int] = []
        fallback = evidence.steps[-1].index if evidence.steps else 0
        valid = {step.index for step in evidence.steps if step.launch is None}

        for assertion in evidence.assertions:
            step = assertion.step_index if assertion.step_index in valid else fallback
            if step not in steps:
                steps.append(step)

        return tuple(steps)

    def __valid_script(self, *, result: GenerationResult) -> bool:
        """
        Return whether a fallback script satisfies the configured dialect checker.
        """

        report = self.__graph_context.dialect.checker.check(text=result.text)
        if not report.issues:
            return True

        logger.info(
            "composed step drafts failed dialect review; falling back to baseline",
            extra={
                "event": "authoring.step_drafts.rejected",
                "execution.id": self.__execution_id,
                "script.issue_codes": [issue.code.value for issue in report.issues],
            },
        )
        return False

    @staticmethod
    def __evidence_for_outcome(*, evidence: Evidence, run_outcome: RunOutcome) -> Evidence:
        """
        Return evidence annotated with the terminal outcome used for authoring and policy.
        """

        if run_outcome is RunOutcome.COMPLETED:
            return evidence.model_copy(update={"outcome": run_outcome})

        reason = evidence.reason or f"Run ended with outcome '{run_outcome.value}'."
        return evidence.model_copy(
            update={
                "partial": True,
                "reason": reason,
                "outcome": run_outcome,
            }
        )

    async def __baseline_for_authoring(self) -> Optional[AuthoringBaseline]:
        """
        Return the deterministic baseline scaffold for final authoring when available.
        """

        artifact = await self.__graph_context.history.peek_baseline_outcome()

        if artifact.metadata.status is not ScriptStatus.GENERATED:
            logger.info(
                "baseline unavailable for final authoring",
                extra={
                    "event": "authoring.run.baseline_unavailable",
                    "execution.id": self.__execution_id,
                    "script.issue_codes": [issue.code.value for issue in artifact.metadata.issues],
                },
            )
            return None

        if not artifact.text or not artifact.text.strip():
            return None

        return AuthoringBaseline(
            content=artifact.text,
            reason=artifact.metadata.review.reason,
            partial=artifact.metadata.review.partial,
            commands=tuple(
                AuthoringBaselineCommand(
                    text=command.text,
                    role=command.role,
                    structural=command.structural,
                    source_steps=command.source_steps,
                )
                for command in artifact.metadata.review.commands
            ),
        )

    async def __cancelled_quality_script(
        self,
        *,
        timeout: float,
        started_at: float,
        assembler: "IntentStrategy.__ResultAssembler",
        script_task: "asyncio.Task[Optional[GenerationResult]]",
    ) -> Optional[GenerationResult]:
        """
        Await cancelled-run quality generation and return None when baseline fallback should run.
        """

        try:
            return await script_task
        except FinalizationTimeoutError as exception:
            assembler.record_finalization_failure(phase=exception.phase)
            logger.warning(
                "cancelled-run quality script finalization timed out; falling back to baseline",
                extra={
                    "event": "fathom.finalization.history.script.cancelled.quality_timed_out",
                    "timeout": timeout,
                    "workflow.id": self.__workflow_id,
                    "duration": time.perf_counter() - started_at,
                    "phase": "fathom.finalization.history.script",
                },
            )
            return None
        except asyncio.CancelledError:
            raise
        except Exception as exception:
            logger.warning(
                "cancelled-run quality script finalization failed; falling back to baseline",
                extra={
                    "event": "fathom.finalization.history.script.cancelled.quality_failed",
                    "workflow.id": self.__workflow_id,
                    "exception.message": str(exception),
                    "exception.type": type(exception).__name__,
                    "duration": time.perf_counter() - started_at,
                    "phase": "fathom.finalization.history.script",
                },
            )
            return None

    async def __stop_cancelled_script_heartbeat(
        self,
        *,
        heartbeat_task: "asyncio.Task[None]",
    ) -> None:
        """
        Stop the cancelled-run heartbeat task after quality generation finishes or falls back.
        """

        heartbeat_task.cancel()
        try:  # noqa: SIM105 - keep cancellation handling explicit for finalization auditability.
            await heartbeat_task
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception(
                "cancelled-run script heartbeat task failed during shutdown",
                extra={
                    "event": "fathom.finalization.history.script.heartbeat.failed",
                    "workflow.id": self.__workflow_id,
                },
            )

    async def __deliver_cancelled_script(
        self,
        *,
        started_at: float,
        quality: Optional[GenerationResult],
        assembler: "IntentStrategy.__ResultAssembler",
    ) -> None:
        """
        Deliver a cancelled-run script through the shared quality-to-baseline finalization path.
        """

        try:
            await self.__deliver_final_script(
                quality=quality,
                run_outcome=RunOutcome.CANCELLED,
            )
        except Exception as exception:
            assembler.record_finalization_failure(phase="fathom.finalization.history.script")
            logger.exception(
                "cancelled-run script fallback failed",
                extra={
                    "event": "fathom.finalization.history.script.cancelled.failed",
                    "workflow.id": self.__workflow_id,
                    "exception.message": str(exception),
                    "exception.type": type(exception).__name__,
                    "duration": time.perf_counter() - started_at,
                    "phase": "fathom.finalization.history.script",
                },
            )
            await self.__emit_script_generation_failed_event(
                metadata=self.__baseline_failure(
                    message=f"Cancelled-run script fallback failed: {exception}"
                ),
                run_outcome=RunOutcome.CANCELLED,
            )

    async def __await_finalization_phase(
        self,
        *,
        phase: str,
        timeout: float,
        awaitable: Awaitable[_FinalizationResult],
    ) -> _FinalizationResult:
        """
        Await one finalization phase under its typed timeout and preserve failures for the caller.
        """

        return await BoundedPhase[_FinalizationResult](
            phase=phase,
            timeout=timeout,
            workflow_id=self.__workflow_id,
        ).execute(awaitable=awaitable)

    async def __emit_cancelled_script_heartbeats(
        self,
        *,
        script_task: "asyncio.Task[Any]",
    ) -> None:
        """
        Emit bounded client heartbeats while cancelled-run script generation is still running.

        The script task has its own finalization timeout, currently shorter than the
        heartbeat ceiling; this loop exits on task completion and is cancelled in
        the publisher's ``finally`` so heartbeat can never outlive script finalization.
        """

        heartbeat = self.__graph_context.configuration.telemetry.phase.heartbeat
        failed_heartbeat_emits = 0

        for _ in range(heartbeat.limit):
            done, _ = await asyncio.wait({script_task}, timeout=heartbeat.threshold)

            if script_task in done:
                return

            try:
                await self.__graph_context.telemetry.info(
                    heartbeat.script_finalization,
                    intent=self.__intent,
                    workflow_id=self.__workflow_id,
                    type=FathomEvent.PHASE_HEARTBEAT,
                    run_outcome=RunOutcome.CANCELLED.value,
                    phase="fathom.finalization.history.script",
                    step=self.__graph_context.agent_state.step_count,
                )
                failed_heartbeat_emits = 0
            except Exception:
                failed_heartbeat_emits += 1
                extra = {
                    "workflow.id": self.__workflow_id,
                    "phase": "fathom.finalization.history.script",
                    "event": "fathom.finalization.history.script.heartbeat.failed",
                    "failure.count": failed_heartbeat_emits,
                }

                if failed_heartbeat_emits == 1:
                    logger.exception("cancelled-run script heartbeat emit failed", extra=extra)
                else:
                    logger.warning("cancelled-run script heartbeat emit failed", extra=extra)

                if failed_heartbeat_emits >= _CANCELLED_SCRIPT_HEARTBEAT_FAILURE_LIMIT:
                    return

    @staticmethod
    def __baseline_failure(*, message: str) -> ScriptFileMetadata:
        """
        Build baseline-sourced failure metadata for finalization failures before an artifact exists.
        """

        return ScriptFileMetadata(
            status=ScriptStatus.FAILED,
            source=ScriptSource.BASELINE,
            issues=(Issue(code=IssueCode.BASELINE_UNAVAILABLE, message=message),),
        )

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
                    "event": "fathom.runner.context.shutdown.failed",
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

    async def __deliver_final_script(
        self,
        *,
        run_outcome: RunOutcome,
        quality: Optional[GenerationResult],
    ) -> None:
        """
        Deliver the completed run's script: quality first, then the deterministic baseline, else a typed failure.
        """

        step = self.__graph_context.agent_state.step_count

        if quality is not None and quality.text.strip():
            selected_event = (
                "script.finalization.quality_selected"
                if quality.source is ScriptSource.QUALITY
                else "script.finalization.step_drafts_selected"
            )
            await self.__graph_context.history.save_script(
                result=quality,
                step_number=step,
                source=quality.source,
            )
            logger.info(
                "finalization selected the authored script",
                extra={
                    "event": selected_event,
                    "script.step": step,
                    "workflow.id": self.__workflow_id,
                    "script.outcome": run_outcome.value,
                    "script.source": quality.source.value,
                    "script.partial": quality.review.partial,
                    "script.line_count": len(quality.text.splitlines()),
                },
            )
            await self.__emit_script_generated_event(
                source=quality.source,
                review=quality.review,
                run_outcome=run_outcome,
                script_data=quality.text,
            )
            return

        logger.info(
            "quality script unavailable; falling back to the baseline",
            extra={
                "event": "script.finalization.quality_unavailable",
                "workflow.id": self.__workflow_id,
                "script.step": step,
                "script.outcome": run_outcome.value,
            },
        )

        artifact = await self.__graph_context.history.read_baseline_outcome(step_number=step)

        if artifact.metadata.status is ScriptStatus.GENERATED and (artifact.text or "").strip():
            logger.info(
                "finalization selected the baseline script",
                extra={
                    "event": "script.finalization.baseline_selected",
                    "workflow.id": self.__workflow_id,
                    "script.step": step,
                    "script.outcome": run_outcome.value,
                    "script.source": ScriptSource.BASELINE.value,
                    "script.line_count": len((artifact.text or "").splitlines()),
                },
            )
            await self.__emit_script_generated_event(
                run_outcome=run_outcome,
                script_data=artifact.text,
                source=ScriptSource.BASELINE,
                review=artifact.metadata.review,
            )
            return

        logger.warning(
            "finalization produced no script; emitting failure",
            extra={
                "event": "script.finalization.failed",
                "workflow.id": self.__workflow_id,
                "script.step": step,
                "script.outcome": run_outcome.value,
                "script.source": artifact.metadata.source.value,
                "script.issue_count": len(artifact.metadata.issues),
                "script.issue_codes": [issue.code.value for issue in artifact.metadata.issues],
            },
        )
        await self.__emit_script_generation_failed_event(
            metadata=artifact.metadata,
            run_outcome=run_outcome,
        )

    async def __emit_script_generated_event(
        self,
        *,
        run_outcome: RunOutcome,
        script_data: Optional[str],
        review: Optional[ScriptReview] = None,
        source: ScriptSource = ScriptSource.QUALITY,
    ) -> None:
        """
        Emit the SCRIPT_GENERATED terminal event whenever the run reaches finalization.

        Carries an ``is_empty`` flag so consumers can distinguish "no script" from "script with content" without parsing the payload.
        ``run_outcome`` tags the event so downstream tooling can correlate empty scripts with operator-aborted runs versus successful completions.
        """

        script_review = review or ScriptReview()
        is_empty_script = not bool(script_data and script_data.strip())
        self.__final_script = None if is_empty_script else script_data

        if is_empty_script:
            logger.warning(
                "final script empty; emitting terminal signal for the client",
                extra={
                    "event": "script.telemetry.empty_generated",
                    "workflow.id": self.__workflow_id,
                    "script.source": source.value,
                    "script.outcome": run_outcome.value,
                    "script.step": self.__graph_context.agent_state.step_count,
                },
            )

        try:
            await self.__graph_context.telemetry.info(
                script_data or "",
                source=source.value,
                is_empty=is_empty_script,
                partial=script_review.partial,
                run_outcome=run_outcome.value,
                workflow_id=self.__workflow_id,
                type=FathomEvent.SCRIPT_GENERATED,
                review_reason=script_review.reason,
                step=self.__graph_context.agent_state.step_count,
                discarded_steps=list(script_review.discarded),
            )
        except Exception as exception:
            self.__log_emit_failed(
                terminal=FathomEvent.SCRIPT_GENERATED, source=source.value, exception=exception
            )
            raise

        logger.info(
            "terminal SCRIPT_GENERATED event emitted",
            extra={
                "event": "script.telemetry.generated_emitted",
                "workflow.id": self.__workflow_id,
                "script.source": source.value,
                "script.is_empty": is_empty_script,
                "script.outcome": run_outcome.value,
                "script.partial": script_review.partial,
                "script.step": self.__graph_context.agent_state.step_count,
            },
        )

    async def __emit_script_generation_failed_event(
        self,
        *,
        run_outcome: RunOutcome,
        metadata: ScriptFileMetadata,
    ) -> None:
        """
        Emit the SCRIPT_GENERATION_FAILED terminal event with structured diagnostics, never an empty success.
        """

        diagnostics = [
            {"code": issue.code.value, "message": issue.message} for issue in metadata.issues
        ]

        try:
            await self.__graph_context.telemetry.info(
                "",
                issues=diagnostics,
                source=metadata.source.value,
                partial=metadata.review.partial,
                review_reason=metadata.review.reason,
                discarded_steps=list(metadata.review.discarded),
                run_outcome=run_outcome.value,
                workflow_id=self.__workflow_id,
                type=FathomEvent.SCRIPT_GENERATION_FAILED,
                step=self.__graph_context.agent_state.step_count,
            )
        except Exception as exception:
            self.__log_emit_failed(
                exception=exception,
                source=metadata.source.value,
                terminal=FathomEvent.SCRIPT_GENERATION_FAILED,
            )
            raise

        logger.info(
            "terminal SCRIPT_GENERATION_FAILED event emitted",
            extra={
                "event": "script.telemetry.failed_emitted",
                "workflow.id": self.__workflow_id,
                "script.source": metadata.source.value,
                "script.outcome": run_outcome.value,
                "script.partial": metadata.review.partial,
                "script.issue_count": len(metadata.issues),
                "script.step": self.__graph_context.agent_state.step_count,
                "script.issue_codes": [issue.code.value for issue in metadata.issues],
            },
        )

    def __log_emit_failed(
        self, *, terminal: FathomEvent, source: str, exception: Exception
    ) -> None:
        """
        Record that emitting a terminal script event to telemetry failed.
        """

        logger.warning(
            "terminal script event emit failed",
            extra={
                "event": "script.telemetry.emit_failed",
                "workflow.id": self.__workflow_id,
                "script.source": source,
                "script.terminal_event": terminal.value,
                "exception.type": type(exception).__name__,
                "exception.message": str(exception),
            },
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
                self.__log_task_cleanup(
                    event="intent.background_task.failed", task_name=task_name, exception=exception
                )
            return

        task.cancel()

        try:
            await task
        except asyncio.CancelledError:
            logger.info(
                "background task cancelled during cleanup",
                extra={
                    "event": "intent.background_task.cancelled",
                    "workflow.id": self.__workflow_id,
                    "task.name": task_name,
                },
            )
        except Exception as exception:
            self.__log_task_cleanup(
                event="intent.background_task.cleanup_failed",
                task_name=task_name,
                exception=exception,
            )

    def __log_task_cleanup(self, *, event: str, task_name: str, exception: Exception) -> None:
        """
        Emit a structured warning for a background task that errored during cleanup.
        """

        logger.warning(
            "background task error during cleanup",
            extra={
                "event": event,
                "workflow.id": self.__workflow_id,
                "task.name": task_name,
                "exception.type": type(exception).__name__,
                "exception.message": str(exception),
            },
        )

    @property
    def step_results(self) -> List[StepResult]:
        """
        Step results accumulated during execution.
        Available after execute() completes.
        """

        return self.__step_results

    @property
    def final_script(self) -> Optional[str]:
        """
        Return the final script content generated during run finalization.
        """

        return self.__final_script

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
                reason = (
                    final_state.values.get(CommonStateKey.COMPLETION_REASON)
                    or final_state.values.get(CommonStateKey.COMPLETION_REASON.value)
                    or final_state.values.get("completion_reason")
                )

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
                failure_diagnostic := (
                    final_state.values.get(CommonStateKey.FAILURE_DIAGNOSTIC)
                    or final_state.values.get(CommonStateKey.FAILURE_DIAGNOSTIC.value)
                    or final_state.values.get("failure_diagnostic")
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
