from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from logging import getLogger
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    cast,
    runtime_checkable,
)

from pydantic import JsonValue

from fathom.adapters.checkpoint import LangGraphPlanStore
from fathom.adapters.signing.noop import NoopSigner
from fathom.base.paths import SharedPathManager
from fathom.base.phase import AbandonablePhase
from fathom.constants import ContextScope, FathomEvent
from fathom.constants.collaboration import (
    ActorKind,
    ArtifactBackend,
    TaskCode,
)
from fathom.constants.qualification import DEFAULT_REJECTION_MESSAGE, RationaleCategory
from fathom.constants.screen import (
    MEMORY_SUMMARY_HASH_PREVIEW_LENGTH,
    MEMORY_SUMMARY_RECENT_LIMIT,
    MEMORY_SUMMARY_UNKNOWN_ACTIVITY,
    KnowledgeKey,
    MemorySummaryKey,
)
from fathom.constants.state import CompletionReason
from fathom.conversation.identity import InteractionIdentity
from fathom.core.capability.catalog import CommandCatalogProvider
from fathom.core.capture.store import CaptureStore
from fathom.core.config.loader import RuntimeConfigLoader
from fathom.core.context.manager import ContextManager
from fathom.core.exceptions import IdentityError, InteractionError
from fathom.core.execution.engine import ExecutionEngine
from fathom.core.services.artifacts import ArtifactCatalog
from fathom.core.services.conversation import ConversationService, Ports
from fathom.core.services.conversation.title import TitleComposer
from fathom.core.services.qualifier.gate import QualificationGatePolicy
from fathom.core.services.recorder import ConversationRecorder
from fathom.core.services.telemetry import PhaseAnnouncer
from fathom.interfaces.device import DevicePort
from fathom.interfaces.interaction import InteractionPort
from fathom.interfaces.knowledge import KnowledgePort
from fathom.interfaces.llm import LLMPort
from fathom.interfaces.memory import MemoryPort
from fathom.interfaces.perception import PerceptionPort
from fathom.interfaces.qualifier import IntentQualifierPort
from fathom.interfaces.signal import SignalPort
from fathom.interfaces.storage import StoragePort
from fathom.interfaces.summarization import SummarizationPort
from fathom.interfaces.telemetry import TelemetryLevel, TelemetryPort
from fathom.runtime.inspection import RuntimeConfigurationInspector
from fathom.schemas.configuration import FathomConfiguration
from fathom.schemas.conversation import ActorInput
from fathom.schemas.exploration import ExplorationGraph
from fathom.schemas.qualification import QualificationVerdict
from fathom.schemas.recording import Completion, Handle, Output, Run, ScriptOutput
from fathom.schemas.results import ExplorationResult, IntentResult
from fathom.schemas.run import Principal, RealignmentPolicy
from fathom.strategies.exploration import ExplorationStrategy
from fathom.strategies.intent import IntentStrategy
from fathom.version import VersionInfo

if TYPE_CHECKING:
    from pathlib import Path

logger = getLogger(__name__)


@runtime_checkable
class TelemetryIdentityUpdater(Protocol):
    """
    Optional telemetry capability for workflow-scoped routing identity.
    """

    def update_identity(self, *, identity: str) -> None:
        """
        Update telemetry routing identity for the active workflow.
        """

        ...


@runtime_checkable
class CancellableStrategy(Protocol):
    """
    Optional strategy capability for cooperative cancellation.
    """

    def cancel(self) -> None:
        """
        Request cooperative cancellation from the strategy.
        """

        ...


@runtime_checkable
class AsyncCloser(Protocol):
    """
    Optional adapter capability for releasing async transport resources.
    """

    async def close(self) -> None:
        """
        Close adapter-owned transport resources.
        """

        ...


class FathomRunner:
    """
    Executes Fathom workflows with configured ports.

    This is the main execution orchestrator that wires together all ports
    and coordinates the execution of automation workflows using hexagonal architecture.

    The runner:
    - Wires ExecutionEngine and ContextManager
    - Manages execution lifecycle
    - Delegates to strategy implementations
    - Returns results compatible with CLI expectations
    """

    def __init__(
        self,
        *,
        llm: LLMPort,
        architect: Optional[LLMPort] = None,
        device: DevicePort,
        memory: MemoryPort,
        signal: SignalPort,
        storage: StoragePort,
        knowledge: KnowledgePort,
        telemetry: TelemetryPort,
        perception: PerceptionPort,
        summarizer: SummarizationPort,
        qualifier: IntentQualifierPort,
        path_manager: SharedPathManager,
        config: Optional[FathomConfiguration] = None,
        interaction: Optional[InteractionPort] = None,
        realignment: Optional[RealignmentPolicy] = None,
        owned_resources: Optional[List[LLMPort]] = None,
        runtime_configuration: Optional[RuntimeConfigLoader] = None,
    ) -> None:
        """
        Initialize runner with all configured ports.
        """

        self.__llm = llm
        self.__architect = architect or llm
        self.__device = device
        self.__perception = perception

        self.__memory = memory
        self.__knowledge = knowledge

        self.__signal = signal
        self.__storage = storage
        self.__telemetry = telemetry
        self.__summarizer = summarizer

        self.__qualifier = qualifier
        self.__interaction = interaction
        self.__config = config or FathomConfiguration()
        self.__runtime_configuration = runtime_configuration

        if self.__runtime_configuration is not None:
            self.__config = self.__config.model_copy(
                update={"oracle": self.__runtime_configuration.oracle()},
            )

        self.__recorder = self.__recorder_for(interaction=interaction)

        self.__path_manager = path_manager

        self.__artifact_catalog = ArtifactCatalog(path_manager=path_manager)

        self.__realignment = realignment or RealignmentPolicy()

        self.__owned_resources: List[LLMPort] = list(owned_resources or [])
        self.__phase = PhaseAnnouncer(
            telemetry=telemetry,
            message=self.__config.telemetry.phase,
        )

        # Wire core components
        self.__catalog = CommandCatalogProvider().build()
        self.__capture_store = CaptureStore()
        self.__engine = ExecutionEngine(
            llm=llm,
            device=device,
            memory=memory,
            signal=signal,
            storage=storage,
            telemetry=telemetry,
            perception=perception,
            path_manager=path_manager,
            catalog=self.__catalog,
            capture_store=self.__capture_store,
        )
        self.__context_manager: Optional[ContextManager] = None

        # Track current workflow for cancellation
        self.__current_strategy: Optional[object] = None

        self.__log_configuration_summary()

    def __log_configuration_summary(self) -> None:
        """
        Emit one structured log line summarizing the runner configuration.
        """

        inspector = RuntimeConfigurationInspector()
        snapshot = inspector.project(
            ports={
                "llm": self.__llm,
                "signal": self.__signal,
                "memory": self.__memory,
                "device": self.__device,
                "storage": self.__storage,
                "knowledge": self.__knowledge,
                "telemetry": self.__telemetry,
                "summarizer": self.__summarizer,
                "perception": self.__perception,
            },
            configuration=self.__config,
            realignment=self.__realignment,
            path_manager=self.__path_manager,
        )
        logger.info(
            "Fathom runner configured",
            extra={
                **snapshot,
                "component": "runtime.runner",
                "package": VersionInfo.payload(),
                "event": "fathom.runner.configured",
                "runtime_configuration_bound": self.__runtime_configuration is not None,
            },
        )

    def __sync_telemetry_identity(self, *, workflow: str) -> None:
        """
        Update telemetry routing identity when the adapter supports it.
        """

        if isinstance(self.__telemetry, TelemetryIdentityUpdater):
            self.__telemetry.update_identity(identity=workflow)

    def __reserve_execution(self, *, execution_id: Optional[str], workflow_id: str) -> str:
        """
        Return the run-owned execution identity, deriving a stable one when absent.
        """

        return execution_id or InteractionIdentity.stable(scope="execution", parts=(workflow_id,))

    @property
    def engine(self) -> ExecutionEngine:
        """
        Get the execution engine.
        """

        return self.__engine

    @property
    def device(self) -> DevicePort:
        """
        Get the device port.
        """

        return self.__device

    @property
    def perception(self) -> PerceptionPort:
        """
        Get the perception port.
        """

        return self.__perception

    @property
    def context(self) -> Optional[ContextManager]:
        """
        Get the context manager.
        """

        return self.__context_manager

    async def run_intent(
        self,
        *,
        intent: str,
        request_id: str,
        principal: Principal,
        max_steps: int = 50,
        use_xml: bool = False,
        execution_id: Optional[str] = None,
        package_name: Optional[str] = None,
        realignment: Optional[RealignmentPolicy] = None,
        context_scope: ContextScope = ContextScope.EXECUTION,
    ) -> IntentResult:
        """
        Execute intent-based workflow.

        All identity is supplied via `principal`; there are no silent fallbacks.
        `request_id` is required and becomes the workflow id used for artifact
        directories and telemetry routing. When recording is enabled, the
        recorder reserves the execution id before strategy execution starts.
        """

        if not request_id:
            raise IdentityError(field="request_id", message="request_id is required")

        start_time = time.time()
        started = datetime.now(tz=timezone.utc)

        workflow_id = request_id

        # The runner owns the execution identity and derives it deterministically
        # from the workflow id, so it is stable across Temporal retries and never
        # depends on the conversation layer. A recording failure can degrade the
        # ledger but can never starve execution of its identity.
        execution_id = self.__reserve_execution(execution_id=execution_id, workflow_id=workflow_id)

        tenant = principal.tenant
        thread = principal.conversation

        responder = principal.agent
        requester = principal.operator
        workspace = principal.workspace

        handle: Optional[Handle] = None
        requested_package = package_name

        self.__sync_telemetry_identity(workflow=workflow_id)

        await self.__telemetry.info(
            "Starting intent workflow",
            intent=intent,
            max_steps=max_steps,
            workflow_id=workflow_id,
            context_scope=context_scope,
        )

        rejection = await self.__qualify_or_reject(
            intent=intent,
            start_time=start_time,
            workflow_id=workflow_id,
        )
        if rejection is not None:
            return rejection

        # Read device state only after the qualifier has allowed the run.
        if not package_name:
            package_name = await self.__device.get_current_package()

        if self.__device.configuration:
            device_serial = self.__device.configuration.identifier
        else:
            device_serial = None

        await self.__telemetry.info(
            "Intent workflow qualified",
            intent=intent,
            max_steps=max_steps,
            workflow_id=workflow_id,
            package_name=package_name,
            device_serial=device_serial,
            context_scope=context_scope,
        )

        if self.__recorder is not None:
            self.__recorder.health.reset()

            handle = await self.__recorder.record_run_started(
                run=Run(
                    tenant=tenant,
                    thread=thread,
                    intent=intent,
                    workspace=workspace,
                    workflow=workflow_id,
                    execution=execution_id,
                    package=requested_package,
                    requester=ActorInput(
                        id=requester,
                        name=requester,
                        kind=ActorKind.HUMAN,
                    ),
                    responder=ActorInput(
                        id=responder,
                        name=responder,
                        kind=ActorKind.AGENT,
                        model=self.__llm_model(),
                        provider=self.__llm_provider(),
                    ),
                    created=started,
                    metadata={
                        "starting_package": package_name,
                        "context_scope": context_scope.value,
                    },
                )
            )

        identity = InteractionIdentity(execution=execution_id)

        # Initialize context namespace. Conversation-scoped runs share memory
        # across runs in the same conversation; execution-scoped runs are
        # isolated by workflow id.
        namespace = thread if context_scope == ContextScope.CONVERSATION else workflow_id

        # Initialize context
        self.__context_manager = ContextManager(memory=self.__memory, workflow_id=namespace)
        self.__context_manager.set_roadmap(intent=intent)

        # Create and execute strategy
        strategy = IntentStrategy(
            intent=intent,
            tenant=tenant,
            thread=thread,
            llm=self.__llm,
            architect=self.__architect,
            requester=requester,
            responder=responder,
            workspace=workspace,
            device=self.__device,
            memory=self.__memory,
            signal=self.__signal,
            storage=self.__storage,
            catalog=self.__catalog,
            workflow_id=workflow_id,
            execution_id=execution_id,
            recorder=self.__recorder,
            package_name=package_name,
            requested_package=requested_package,
            telemetry=self.__telemetry,
            configuration=self.__config,
            summarizer=self.__summarizer,
            perception=self.__perception,
            path_manager=self.__path_manager,
            realignment=realignment or self.__realignment,
            runtime_configuration=self.__runtime_configuration,
            plans=LangGraphPlanStore(),
            max_steps=max_steps or self.__config.intent.max_steps,
            use_xml=use_xml if use_xml is not None else self.__config.intent.use_xml_grounding,
        )
        self.__current_strategy = strategy

        try:
            # Execute strategy
            execution_result = await strategy.execute()

            # Get progress info
            progress = strategy.get_progress()

            # Get subgoal execution audit trail
            executed_subgoals, skipped_subgoals, subgoal_count = (
                strategy.get_subgoal_execution_audit()
            )

            # Collect metrics from strategy - use to_report_dict() for proper format
            strategy_metrics = strategy.get_metrics()
            metrics = strategy_metrics.to_report_dict() if strategy_metrics else {}

            # Bound memory summary so a stuck store read cannot block result delivery.
            raw_memory_summary = await AbandonablePhase(
                workflow_id=workflow_id,
                phase="fathom.runner.memory.summary",
                timeout=self.__config.intent.finalization.runtime.memory_summary,
            ).execute(awaitable=self.__get_memory_summary())

            memory_summary: Dict[str, JsonValue] = raw_memory_summary if raw_memory_summary else {}

            # Build IntentResult
            duration = time.time() - start_time
            is_cancelled = execution_result.is_cancelled

            if is_cancelled:
                error = None
                success = False
                status = "cancelled"
                completion_reason = self.__completion_reason(
                    status=status,
                    fallback=(
                        strategy.completion_reason
                        or str(progress.get("completion_reason") or "").strip()
                        or CompletionReason.CANCELLED.value
                    ),
                )
            else:
                success = execution_result.success
                error = execution_result.error
                status = "completed" if execution_result.success else "failed"
                raw_completion_reason = (
                    strategy.completion_reason
                    or str(progress.get("completion_reason") or "").strip()
                    or execution_result.error
                    or (
                        CompletionReason.SUCCESS.value
                        if execution_result.success
                        else CompletionReason.FAILED.value
                    )
                )
                completion_reason = self.__completion_reason(
                    error=error,
                    status=status,
                    fallback=raw_completion_reason,
                )

            result = IntentResult(
                error=error,
                status=status,
                intent=intent,
                metrics=metrics,
                success=success,
                duration=duration,
                workflow_id=workflow_id,
                subgoal_count=subgoal_count,
                script=strategy.final_script,
                memory_summary=memory_summary,
                skipped_subgoals=skipped_subgoals,
                step_results=strategy.step_results,
                executed_subgoals=executed_subgoals,
                completion_reason=completion_reason,
                steps_taken=progress.get("step_count", 0),
                steps_executed=progress.get("step_count", 0),
            )

            await self.__telemetry.info(
                "Workflow execution finalized",
                duration=duration,
                success=result.success,
                steps_taken=result.steps_taken,
                type=FathomEvent.WORKFLOW_COMPLETED,
            )

            if self.__recorder is not None and handle is not None:
                finished = datetime.now(tz=timezone.utc)

                await self.__recorder.record_run_finished(
                    completion=Completion(
                        handle=handle,
                        status=result.status,
                        success=result.success,
                        reason=result.completion_reason,
                        result=identity.message(name="result"),
                        code=self.__task_code(
                            success=result.success, reason=result.completion_reason
                        ),
                        metadata={},
                        finished=finished,
                        error=result.error,
                        steps=result.steps_taken,
                        elapsed=int(duration * 1000),
                    )
                )
                await self.__record_generated_script(
                    title=intent,
                    handle=handle,
                    created=finished,
                    content=result.script,
                    metadata={"source": "finalization"},
                )
                await self.__record_workflow_artifacts(
                    handle=handle,
                    script_title=intent,
                    package_name=package_name,
                )

            return result

        except Exception as exception:
            await self.__telemetry.warning(
                "Run failed before completion",
                workflow_id=workflow_id,
                error=str(exception),
                error_type=type(exception).__name__,
            )
            if self.__recorder is not None and handle is not None:
                finished = datetime.now(tz=timezone.utc)

                try:
                    progress = strategy.get_progress()
                    steps = int(progress.get("step_count") or 0)

                    await self.__recorder.record_run_failed(
                        completion=Completion(
                            steps=steps,
                            metadata={},
                            handle=handle,
                            success=False,
                            status="failed",
                            finished=finished,
                            error=str(exception),
                            code=TaskCode.UNKNOWN_ERROR,
                            reason=self.__completion_reason(
                                error=str(exception),
                                status=CompletionReason.FAILED.value,
                                fallback=CompletionReason.FAILED.value,
                            ),
                            result=identity.message(name="result"),
                            elapsed=int((time.time() - start_time) * 1000),
                        )
                    )
                    await self.__record_workflow_artifacts(
                        handle=handle,
                        script_title=intent,
                        package_name=package_name,
                    )
                except InteractionError as interaction_exception:
                    await self.__telemetry.warning(
                        "Failed to record interaction failure",
                        workflow_id=workflow_id,
                        error=str(interaction_exception),
                    )

            raise

        finally:
            self.__current_strategy = None

    async def run_exploration(
        self,
        *,
        request_id: str,
        principal: Principal,
        max_steps: int = 100,
        execution_id: Optional[str] = None,
        package_name: Optional[str] = None,
    ) -> ExplorationResult:
        """
        Execute exploration workflow.

        `request_id` is required and becomes the workflow id. When recording is
        enabled, the recorder reserves the execution id before exploration starts.
        """

        if not request_id:
            raise IdentityError(field="request_id", message="request_id is required")

        start_time = time.time()
        started = datetime.now(tz=timezone.utc)

        workflow_id = request_id

        # The runner owns the execution identity and derives it deterministically
        # from the workflow id, so it is stable across Temporal retries and never
        # depends on the conversation layer. A recording failure can degrade the
        # ledger but can never starve execution of its identity.
        execution_id = self.__reserve_execution(execution_id=execution_id, workflow_id=workflow_id)

        tenant = principal.tenant
        thread = principal.conversation

        responder = principal.agent
        requester = principal.operator
        workspace = principal.workspace

        intent = "Explore application structure"
        handle: Optional[Handle] = None

        self.__sync_telemetry_identity(workflow=workflow_id)

        # Use provided package name or fetch from device
        if not package_name:
            package_name = await self.__device.get_current_package()

        if self.__device.configuration:
            device_serial = self.__device.configuration.identifier
        else:
            device_serial = None

        await self.__telemetry.info(
            "Starting exploration workflow",
            max_steps=max_steps,
            workflow_id=workflow_id,
            package_name=package_name,
            device_serial=device_serial,
        )

        if self.__recorder is not None:
            self.__recorder.health.reset()

            handle = await self.__recorder.record_run_started(
                run=Run(
                    intent=intent,
                    tenant=tenant,
                    thread=thread,
                    workspace=workspace,
                    workflow=workflow_id,
                    package=package_name,
                    execution=execution_id,
                    requester=ActorInput(
                        id=requester,
                        name=requester,
                        kind=ActorKind.HUMAN,
                    ),
                    responder=ActorInput(
                        id=responder,
                        name=responder,
                        kind=ActorKind.AGENT,
                        model=self.__llm_model(),
                        provider=self.__llm_provider(),
                    ),
                    created=started,
                    metadata={"mode": "exploration"},
                )
            )

        identity = InteractionIdentity(execution=execution_id)

        # Initialize context
        self.__context_manager = ContextManager(memory=self.__memory, workflow_id=workflow_id)
        self.__context_manager.set_roadmap(intent=intent)

        strategy = ExplorationStrategy(
            tenant=tenant,
            thread=thread,
            llm=self.__llm,
            requester=requester,
            responder=responder,
            device=self.__device,
            memory=self.__memory,
            signal=self.__signal,
            storage=self.__storage,
            workflow_id=workflow_id,
            execution_id=execution_id,
            package_name=package_name,
            telemetry=self.__telemetry,
            configuration=self.__config,
            perception=self.__perception,
            path_manager=self.__path_manager,
            seed=self.__config.exploration.random_seed,
            timeout=float(self.__config.exploration.timeout),
            runtime_configuration=self.__runtime_configuration,
            max_steps=max_steps or self.__config.exploration.max_steps,
        )

        self.__current_strategy = strategy

        try:
            # Execute strategy
            execution_result = await strategy.execute()

            # Get progress info
            progress = strategy.get_progress()
            stats = progress.get("stats", {})

            # Extract discovered activities from graph
            graph = strategy.graph
            discovered_activities = list({node.activity for node in graph.nodes.values()})

            # Calculate coverage (percentage of screens explored vs total discovered)
            unexplored = stats.get("unexplored", 0)
            unique_screens = stats.get("unique_screens", 0)

            coverage_percentage = (
                ((unique_screens - unexplored) / unique_screens * 100.0)
                if unique_screens > 0
                else 0.0
            )

            # Export graph structure
            screen_graph = await self.__export_graph(graph=graph)

            # Build ExplorationResult
            duration = time.time() - start_time

            result = ExplorationResult(
                duration=duration,
                workflow_id=workflow_id,
                screen_graph=screen_graph,
                error=execution_result.error,
                unique_screens=unique_screens,
                success=execution_result.success,
                coverage_percentage=coverage_percentage,
                completion_reason="Exploration completed",
                discovered_activities=discovered_activities,
                total_actions=stats.get("total_actions", 0),
                steps_executed=progress.get("steps", 0),
                total_transitions=stats.get("total_transitions", 0),
                status="completed" if execution_result.success else "failed",
            )

            await self.__telemetry.info(
                "Exploration workflow completed",
                duration=duration,
                total_actions=result.total_actions,
                unique_screens=result.unique_screens,
            )

            if self.__recorder is not None and handle is not None:
                finished = datetime.now(tz=timezone.utc)

                await self.__recorder.record_run_finished(
                    completion=Completion(
                        handle=handle,
                        finished=finished,
                        error=result.error,
                        status=result.status,
                        success=result.success,
                        steps=result.steps_executed,
                        elapsed=int(duration * 1000),
                        reason=result.completion_reason,
                        result=identity.message(name="result"),
                        metadata={"mode": "exploration"},
                        code=self.__task_code(
                            success=result.success, reason=result.completion_reason
                        ),
                    )
                )
                await self.__record_workflow_artifacts(
                    handle=handle,
                    package_name=package_name,
                    script_title="Exploration script",
                )

            return result

        except Exception as exception:
            await self.__telemetry.warning(
                "Exploration run failed before completion",
                workflow_id=workflow_id,
                error=str(exception),
                error_type=type(exception).__name__,
            )
            if self.__recorder is not None and handle is not None:
                finished = datetime.now(tz=timezone.utc)

                try:
                    progress = strategy.get_progress()
                    steps = int(progress.get("steps") or 0)

                    await self.__recorder.record_run_failed(
                        completion=Completion(
                            steps=steps,
                            handle=handle,
                            success=False,
                            status="failed",
                            finished=finished,
                            error=str(exception),
                            code=TaskCode.UNKNOWN_ERROR,
                            reason=CompletionReason.FAILED.value,
                            result=identity.message(name="result"),
                            elapsed=int((time.time() - start_time) * 1000),
                            metadata={"mode": "exploration"},
                        )
                    )
                    await self.__record_workflow_artifacts(
                        handle=handle,
                        package_name=package_name,
                        script_title="Exploration script",
                    )

                except InteractionError as interaction_exception:
                    await self.__telemetry.warning(
                        "Failed to record exploration failure",
                        workflow_id=workflow_id,
                        error=str(interaction_exception),
                    )

            raise

        finally:
            self.__current_strategy = None

    def cancel(self) -> None:
        """
        Cancel the currently running workflow.
        """

        if self.__current_strategy:
            logger.warning("Workflow cancellation requested")

            if isinstance(self.__current_strategy, CancellableStrategy):
                self.__current_strategy.cancel()
            else:
                logger.warning("Strategy does not support cancellation")

    async def notify(
        self,
        *,
        message: str,
        level: TelemetryLevel,
        event_type: Optional[FathomEvent] = None,
    ) -> None:
        """
        Emit a client-facing telemetry message with an optional event type.
        """

        context: Dict[str, FathomEvent] = {}

        if event_type is not None:
            context["type"] = event_type

        if level == "debug":
            await self.__telemetry.debug(message, **context)

        elif level == "info":
            await self.__telemetry.info(message, **context)

        elif level == "warning":
            await self.__telemetry.warning(message, **context)

        else:
            await self.__telemetry.error(message, **context)

    async def cleanup(self) -> None:
        """
        Cleanup all resources held by the runner and its ports.
        """

        # 0. Phase announcer — stop any heartbeat/background telemetry.
        try:
            await self.__phase.shutdown()
        except Exception as exception:
            logger.warning(f"[FathomRunner] phase shutdown failed: {exception}")

        # 1. Context manager — drain persist queue, cancel background tasks
        if self.__context_manager is not None:
            try:
                await self.__context_manager.shutdown()
            except Exception as exception:
                logger.warning(f"[FathomRunner] context_manager shutdown failed: {exception}")

        if self.__recorder is not None:
            try:
                await self.__recorder.drain_background_tasks()
            except Exception as exception:
                logger.warning(f"[FathomRunner] recorder drain failed: {exception}")

        # 2. LLM — delete cached content, close clients
        try:
            await self.__llm.cleanup()
        except Exception as exception:
            logger.warning(f"[FathomRunner] llm cleanup failed: {exception}")

        for resource in self.__owned_resources:
            try:
                await resource.cleanup()
            except Exception as exception:
                logger.warning(f"[FathomRunner] owned resource cleanup failed: {exception}")

        # 3. Device — close HTTP client (ADB remote, iOS remote)
        if isinstance(self.__device, AsyncCloser):
            try:
                await self.__device.close()
            except Exception as exception:
                logger.warning(f"[FathomRunner] device close failed: {exception}")

        # 4. Telemetry — close Redis connection if applicable
        if isinstance(self.__telemetry, AsyncCloser):
            try:
                await self.__telemetry.close()
            except Exception as exception:
                logger.warning(f"[FathomRunner] telemetry close failed: {exception}")

        # 5. Interaction — close durable conversation pools.
        if self.__interaction is not None:
            try:
                await self.__interaction.aclose()
            except Exception as exception:
                logger.warning(f"[FathomRunner] interaction close failed: {exception}")

        logger.info("[FathomRunner] cleanup completed")

    async def __qualify_or_reject(
        self,
        *,
        intent: str,
        workflow_id: str,
        start_time: float,
    ) -> Optional[IntentResult]:
        """
        Qualify the intent before any device interaction.
        """

        logger.info(
            "[FathomRunner] Qualifying intent before device interaction",
            extra={"workflow_id": workflow_id, "intent_length": len(intent or "")},
        )

        qualifier_started_at = time.perf_counter()
        await self.__phase.intent_qualifying(intent=intent)

        try:
            verdict = await self.__qualifier.qualify(intent=intent)
        finally:
            await self.__phase.shutdown()

        qualifier_latency = time.perf_counter() - qualifier_started_at
        gate_policy = QualificationGatePolicy(configuration=self.__config.qualifier)
        blocked = gate_policy.should_block(verdict=verdict)

        verdict_log_extra: Dict[str, Any] = {
            "blocked": blocked,
            "workflow_id": workflow_id,
            "label": verdict.label.value,
            "latency": qualifier_latency,
            "confidence": verdict.confidence,
            "category": verdict.rationale.category.value,
        }
        if verdict.rationale.category == RationaleCategory.QUALIFIER_ERROR:
            verdict_log_extra["reasoning"] = verdict.rationale.reasoning

        logger.info("[FathomRunner] Qualifier returned verdict", extra=verdict_log_extra)

        verdict_payload: Dict[str, Any] = {
            "intent": intent,
            "workflow_id": workflow_id,
            "latency": qualifier_latency,
            "label": verdict.label.value,
            "confidence": verdict.confidence,
            "rationale": verdict.rationale.model_dump(),
        }

        if blocked:
            logger.warning(
                "[FathomRunner] Qualifier blocked the intent",
                extra={
                    "workflow_id": workflow_id,
                    "label": verdict.label.value,
                    "confidence": verdict.confidence,
                    "category": verdict.rationale.category.value,
                },
            )
            return await self.__build_rejection_result(
                intent=intent,
                verdict=verdict,
                start_time=start_time,
                workflow_id=workflow_id,
                verdict_payload=verdict_payload,
            )

        await self.__telemetry.info(
            "Got it, getting started...",
            type=FathomEvent.INTENT_QUALIFIED,
            **verdict_payload,
        )
        logger.info(
            "[FathomRunner] Qualifier allowed the intent, proceeding to execution",
            extra={
                "workflow_id": workflow_id,
                "latency": qualifier_latency,
                "label": verdict.label.value,
                "confidence": verdict.confidence,
                "category": verdict.rationale.category.value,
            },
        )
        return None

    async def __build_rejection_result(
        self,
        *,
        intent: str,
        workflow_id: str,
        start_time: float,
        verdict: QualificationVerdict,
        verdict_payload: Dict[str, Any],
    ) -> IntentResult:
        """
        Emit the rejection notification and build a terminal result.
        """

        user_message = verdict.message or DEFAULT_REJECTION_MESSAGE
        duration = time.time() - start_time

        await self.__telemetry.warning(
            user_message,
            type=FathomEvent.INTENT_REJECTED,
            **verdict_payload,
        )
        await self.__telemetry.info(
            "All wrapped up.",
            success=False,
            steps_taken=0,
            duration=duration,
            type=FathomEvent.WORKFLOW_COMPLETED,
        )

        try:
            await self.__phase.shutdown()
        except Exception as exception:
            logger.warning(
                "Runner phase shutdown after qualifier rejection failed: %s",
                exception,
            )

        return IntentResult(
            error=None,
            metrics={},
            intent=intent,
            success=False,
            steps_taken=0,
            subgoal_count=0,
            step_results=[],
            steps_executed=0,
            status="completed",
            memory_summary={},
            duration=duration,
            skipped_subgoals=[],
            executed_subgoals=[],
            workflow_id=workflow_id,
            completion_reason=CompletionReason.NOT_EXECUTABLE.value,
        )

    async def __record_workflow_artifacts(
        self,
        *,
        handle: Handle,
        package_name: str,
        script_title: str,
    ) -> None:
        """
        Record workflow-scope artifacts (history, script) into the timeline.

        Per-step artifacts (screenshots, traces, xml, annotated) are recorded
        from the graph node boundary inside the strategy, so they reach the
        database as soon as the step finishes. Anything without a step prefix
        in its filename is owned by the workflow and persisted here at the end
        of the run.
        """

        if self.__recorder is None:
            return

        identity = InteractionIdentity(execution=handle.execution)

        logger.info(
            "Recording workflow artifacts",
            extra={
                "event": "conversation.artifacts.workflow.started",
                "workflow.id": handle.workflow,
                "conversation.id": handle.thread,
            },
        )
        artifacts = await self.__artifact_catalog.discover(
            workflow=handle.workflow,
            only_workflow_scope=True,
            package_name=package_name,
        )

        recorded = 0
        for path, stat in artifacts:
            captured_at = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)

            metadata: Dict[str, JsonValue] = {
                "filename": path.name,
                "captured_at": captured_at.isoformat(),
                "category": self.__artifact_catalog.category(path=path),
            }

            resolved_kind = self.__artifact_catalog.kind(path=path)

            if resolved_kind is not None:
                try:
                    await self.__recorder.record_artifact(
                        output=Output(
                            uri=str(path),
                            task=handle.task,
                            size=stat.st_size,
                            metadata=metadata,
                            kind=resolved_kind,
                            created=captured_at,
                            tenant=handle.tenant,
                            thread=handle.thread,
                            actor=handle.responder,
                            workflow=handle.workflow,
                            execution=handle.execution,
                            workspace=handle.workspace,
                            backend=ArtifactBackend.LOCAL,
                            id=identity.artifact(path=path),
                            mime=self.__artifact_catalog.mime(path=path),
                            retention=self.__artifact_catalog.retention(path=path),
                        )
                    )
                    recorded += 1
                except Exception:
                    logger.exception(
                        "Workflow artifact recording failed",
                        extra={
                            "event": "conversation.artifacts.workflow.failed",
                            "path": str(path),
                            "workflow.id": handle.workflow,
                            "conversation.id": handle.thread,
                        },
                    )
            else:
                logger.warning(
                    "Skipping unclassified workflow artifact",
                    extra={
                        "event": "conversation.artifacts.workflow.skipped",
                        "path": str(path),
                        "workflow.id": handle.workflow,
                        "conversation.id": handle.thread,
                    },
                )
            if self.__artifact_catalog.is_script(path=path):
                await self.__record_script_content(
                    path=path,
                    handle=handle,
                    task=handle.task,
                    metadata=metadata,
                    title=script_title,
                    created=captured_at,
                )
        logger.info(
            "Recorded workflow artifacts",
            extra={
                "event": "conversation.artifacts.workflow.completed",
                "workflow.id": handle.workflow,
                "artifacts.recorded": recorded,
                "conversation.id": handle.thread,
                "artifacts.discovered": len(artifacts),
            },
        )

    async def __record_script_content(
        self,
        *,
        task: str,
        title: str,
        path: Path,
        handle: Handle,
        created: datetime,
        metadata: Dict[str, JsonValue],
    ) -> None:
        """
        Persist generated script text as an editable script domain record.
        """

        if self.__recorder is None:
            return

        try:
            content = await asyncio.to_thread(path.read_text, encoding="utf-8")
        except FileNotFoundError:
            return

        await self.__record_generated_script(
            task=task,
            title=title,
            handle=handle,
            content=content,
            created=created,
            script_name=str(path),
            metadata={**metadata, "uri": str(path)},
        )

    async def __record_generated_script(
        self,
        *,
        title: str,
        handle: Handle,
        created: datetime,
        content: Optional[str],
        metadata: Dict[str, JsonValue],
        task: Optional[str] = None,
        script_name: str = "final",
    ) -> None:
        """
        Persist generated script text without depending on artifact file discovery.
        """

        if self.__recorder is None or content is None or not content.strip():
            return

        identity = InteractionIdentity(execution=handle.execution)

        logger.info(
            "Recording generated script",
            extra={
                "event": "conversation.script.record.started",
                "workflow.id": handle.workflow,
                "script.name": script_name,
                "conversation.id": handle.thread,
            },
        )
        await self.__recorder.record_script(
            output=ScriptOutput(
                title=title,
                artifact=None,
                content=content,
                created=created,
                metadata=metadata,
                tenant=handle.tenant,
                thread=handle.thread,
                actor=handle.responder,
                task=task or handle.task,
                workflow=handle.workflow,
                execution=handle.execution,
                workspace=handle.workspace,
                summary="Generated script export.",
                id=identity.script(name=script_name),
            )
        )
        logger.info(
            "Recorded generated script",
            extra={
                "event": "conversation.script.record.completed",
                "workflow.id": handle.workflow,
                "conversation.id": handle.thread,
                "script.name": script_name,
            },
        )

    def __recorder_for(
        self, *, interaction: Optional[InteractionPort]
    ) -> Optional[ConversationRecorder]:
        """
        Build the optional conversation recorder from the interaction port.
        """

        if interaction is None:
            return None

        return ConversationRecorder(
            telemetry=self.__telemetry,
            conversation=ConversationService(
                signer=NoopSigner(),
                ports=self.__ports(interaction=interaction),
            ),
            title=TitleComposer(llm=self.__llm),
        )

    def __ports(self, *, interaction: InteractionPort) -> Ports:
        """
        Map the configured interaction adapter into conversation service ports.
        """

        return Ports(interaction=interaction)

    def __completion_reason(
        self,
        *,
        status: str,
        fallback: str,
        error: Optional[str] = None,
    ) -> str:
        """
        Return the recorded completion reason without synthetic terminal prose.
        """

        return (error or fallback or status).strip() or status

    def __task_code(self, *, success: bool, reason: str) -> TaskCode:
        """
        Resolve the terminal task code for a workflow result.
        """

        if success:
            return TaskCode.COMPLETED

        normalized = reason.casefold().strip()

        if normalized in {
            CompletionReason.CANCELLED.value.casefold(),
            CompletionReason.OPERATOR_ABORTED.value.casefold(),
        }:
            return TaskCode.USER_CANCELLED

        if normalized == CompletionReason.MAX_STEPS.value.casefold():
            return TaskCode.TIMEOUT

        return TaskCode.UNKNOWN_ERROR

    def __llm_provider(self) -> str:
        """
        Derive a stable provider hint from the configured LLM adapter.
        """

        provider = self.__llm.__class__.__name__.removesuffix("LLM").lower()
        return provider or "unknown"

    def __llm_model(self) -> Optional[str]:
        """
        Return the configured model name when the adapter exposes one.
        """

        model = self.__llm.model_name
        if isinstance(model, str) and model:
            return model

        return None

    async def __get_memory_summary(self) -> Dict[str, JsonValue]:
        """
        Get memory summary from memory port.
        """

        try:
            knowledge = await self.__memory.get_all_knowledge()
            raw_screens = knowledge.get(KnowledgeKey.SCREENS, [])
            screens: List[JsonValue] = list(raw_screens) if isinstance(raw_screens, list) else []

            recent: List[JsonValue] = []
            for screen in screens[:MEMORY_SUMMARY_RECENT_LIMIT]:
                if not isinstance(screen, dict):
                    continue
                hashed = str(screen.get(KnowledgeKey.HASH) or "")[
                    :MEMORY_SUMMARY_HASH_PREVIEW_LENGTH
                ]
                recent.append(
                    {
                        KnowledgeKey.HASH.value: hashed,
                        KnowledgeKey.DESCRIPTION.value: str(
                            screen.get(KnowledgeKey.DESCRIPTION) or ""
                        ),
                        KnowledgeKey.ACTIVITY.value: str(
                            screen.get(KnowledgeKey.ACTIVITY) or MEMORY_SUMMARY_UNKNOWN_ACTIVITY
                        ),
                    }
                )

            experience = knowledge.get(KnowledgeKey.EXPERIENCE_COUNT, 0)
            experience_count = experience if isinstance(experience, int) else 0

            return {
                MemorySummaryKey.SCREENS.value: recent,
                MemorySummaryKey.TOTAL_SCREENS.value: len(screens),
                MemorySummaryKey.EXPERIENCE_COUNT.value: experience_count,
            }
        except Exception as exception:
            await self.__telemetry.warning(f"Failed to get memory summary: {exception}")
            return {
                MemorySummaryKey.SCREENS.value: [],
                MemorySummaryKey.TOTAL_SCREENS.value: 0,
                MemorySummaryKey.EXPERIENCE_COUNT.value: 0,
            }

    async def __export_graph(self, graph: ExplorationGraph) -> Dict[str, JsonValue]:
        """
        Export exploration graph to dictionary.
        """

        try:
            nodes: Dict[str, JsonValue] = {}

            for fingerprint, node in graph.nodes.items():
                actions: List[JsonValue] = list(node.actions)
                transitions: Dict[str, JsonValue] = dict(node.transitions)
                nodes[fingerprint] = {
                    "actions": actions,
                    "visits": node.visits,
                    "activity": node.activity,
                    "transitions": transitions,
                }

            edges: List[JsonValue] = [
                {"origin": origin, "action": action, "destination": dest}
                for origin, action, dest in graph.edges
            ]
            stats = cast("Dict[str, JsonValue]", graph.get_stats())

            return {
                "nodes": nodes,
                "edges": edges,
                "stats": stats,
            }
        except Exception as exception:
            await self.__telemetry.warning(f"Failed to export graph: {exception}")
            return {}
