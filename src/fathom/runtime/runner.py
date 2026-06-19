from __future__ import annotations

import time
import uuid
from logging import getLogger
from typing import Any, Dict, List, Optional, cast

from fathom.base.paths import SharedPathManager
from fathom.base.phase import AbandonablePhase
from fathom.constants import ContextScope, FathomEvent
from fathom.constants.finalization import FinalizationPhase
from fathom.constants.qualification import DEFAULT_REJECTION_MESSAGE, RationaleCategory
from fathom.constants.state import CompletionReason
from fathom.core.config.loader import RuntimeConfigLoader
from fathom.core.context.manager import ContextManager
from fathom.core.execution.engine import ExecutionEngine
from fathom.core.services.qualifier.gate import QualificationGatePolicy
from fathom.core.services.telemetry import PhaseAnnouncer
from fathom.interfaces.device import DevicePort
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
from fathom.schemas.exploration import ExplorationGraph
from fathom.schemas.qualification import QualificationVerdict
from fathom.schemas.results import ExplorationResult, IntentResult
from fathom.schemas.run import RealignmentPolicy
from fathom.strategies.exploration import ExplorationStrategy
from fathom.strategies.intent import IntentStrategy
from fathom.version import VersionInfo

logger = getLogger(__name__)


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
        device: DevicePort,
        perception: PerceptionPort,
        memory: MemoryPort,
        signal: SignalPort,
        storage: StoragePort,
        knowledge: KnowledgePort,
        telemetry: TelemetryPort,
        summarizer: SummarizationPort,
        qualifier: IntentQualifierPort,
        path_manager: SharedPathManager,
        config: Optional[FathomConfiguration] = None,
        realignment: Optional[RealignmentPolicy] = None,
        runtime_configuration: Optional[RuntimeConfigLoader] = None,
        owned_resources: Optional[List[LLMPort]] = None,
    ) -> None:
        """
        Initialize runner with all configured ports.

        ``runtime_configuration`` is the application-layer translator that the
        caller pre-bound to its own :class:`FathomSettings` (in the
        Temporal worker registry, service bridges, the CLI, ...).
        It is propagated to :class:`IntentStrategy` and
        :class:`ExplorationStrategy` so :class:`AdapterAssembly`
        observes the same settings the caller built — not a fresh
        ``FathomSettings()`` that only resolves fathom's own env
        aliases and silently misses deployment-prefixed names like
        ``DRIZZ_GOOGLE_APPLICATION_CREDENTIALS_JSON``.

        ``owned_resources``: optional list of LLM ports the runner takes
        ownership of for cleanup purposes. Used by SDK callers that construct
        the runner via FathomBuilder with .with_assembly(...) — the builder
        creates a dedicated qualifier LLM internally and hands it to the
        runner so the caller doesn't have to track it separately. Temporal /
        CLI callers use RunnerComposition at the composition root instead and
        pass nothing here.

        Hexagonal note: we deliberately accept the *loader*
        (Application layer) for runtime_configuration, never the raw
        :class:`FathomSettings` (Infrastructure). The loader keeps the
        settings as a private reference; nothing in the runner /
        strategy can hand the credentials material to a logger or out
        as a Temporal activity argument.
        """

        self.__llm = llm
        self.__device = device
        self.__perception = perception

        self.__memory = memory
        self.__knowledge = knowledge

        self.__signal = signal
        self.__storage = storage
        self.__telemetry = telemetry
        self.__qualifier = qualifier
        self.__summarizer = summarizer
        self.__runtime_configuration = runtime_configuration
        self.__path_manager = path_manager
        self.__config = config or FathomConfiguration()
        self.__realignment = realignment or RealignmentPolicy()
        self.__owned_resources: List[LLMPort] = list(owned_resources or [])
        self.__phase = PhaseAnnouncer(
            telemetry=telemetry,
            message=self.__config.telemetry.phase,
        )

        # Wire core components
        self.__engine = ExecutionEngine(
            llm=llm,
            device=device,
            perception=perception,
            memory=memory,
            signal=signal,
            storage=storage,
            telemetry=telemetry,
            path_manager=path_manager,
        )
        self.__context_manager: Optional[ContextManager] = None

        # Track current workflow for cancellation
        self.__current_strategy: Optional[object] = None

        self.__log_configuration_summary()

    def __log_configuration_summary(self) -> None:
        """
        Emit a single structured log line summarizing the wired configuration.
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
        intent: str,
        max_steps: int = 50,
        use_xml: bool = False,
        request_id: Optional[str] = None,
        package_name: Optional[str] = None,
        conversation_id: Optional[str] = None,
        realignment: Optional[RealignmentPolicy] = None,
        context_scope: ContextScope = ContextScope.EXECUTION,
    ) -> IntentResult:
        """
        Execute intent-based workflow.
        """

        start_time = time.time()
        workflow_id = request_id or uuid.uuid4().hex[:8]

        # Synchronize telemetry identity with workflow_id for routing
        # update_identity is available on some telemetry implementations
        telemetry_with_identity = cast("Any", self.__telemetry)
        if hasattr(telemetry_with_identity, "update_identity"):
            telemetry_with_identity.update_identity(identity=workflow_id)

        await self.__telemetry.info(
            "Starting intent workflow",
            intent=intent,
            max_steps=max_steps,
            workflow_id=workflow_id,
            context_scope=context_scope,
        )

        rejection = await self.__qualify_or_reject(
            intent=intent, workflow_id=workflow_id, start_time=start_time
        )
        if rejection is not None:
            return rejection

        # Read device state only after the gate has allowed the run.
        if not package_name:
            package_name = await self.__device.get_current_package()

        # Initialize context namespace
        namespace = (
            conversation_id
            if context_scope == ContextScope.CONVERSATION and conversation_id
            else workflow_id
        )

        # Initialize context
        self.__context_manager = ContextManager(memory=self.__memory, workflow_id=namespace)
        self.__context_manager.set_roadmap(intent=intent)

        # Create and execute strategy
        strategy = IntentStrategy(
            intent=intent,
            llm=self.__llm,
            device=self.__device,
            memory=self.__memory,
            signal=self.__signal,
            storage=self.__storage,
            workflow_id=workflow_id,
            package_name=package_name,
            telemetry=self.__telemetry,
            configuration=self.__config,
            summarizer=self.__summarizer,
            perception=self.__perception,
            path_manager=self.__path_manager,
            realignment=realignment or self.__realignment,
            max_steps=max_steps or self.__config.intent.max_steps,
            runtime_configuration=self.__runtime_configuration,
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

            # Get memory summary under a bounded abandonable phase so a stuck SQLite read
            # never gates result delivery.
            raw_memory_summary = await AbandonablePhase(
                phase=FinalizationPhase.MEMORY_SUMMARY,
                timeout=self.__config.intent.finalization.runtime.memory_summary,
                workflow_id=workflow_id,
            ).execute(awaitable=self.__get_memory_summary())
            memory_summary: Dict[str, Any] = raw_memory_summary if raw_memory_summary else {}

            # Build IntentResult
            duration = time.time() - start_time
            is_cancelled = execution_result.is_cancelled

            if is_cancelled:
                error = None
                success = False
                status = "failed"
                completion_reason = CompletionReason.CANCELLED.value
            else:
                success = execution_result.success
                error = execution_result.error
                status = "completed" if execution_result.success else "failed"
                completion_reason = (
                    strategy.completion_reason
                    or str(progress.get("completion_reason") or "").strip()
                    or execution_result.error
                    or (
                        CompletionReason.SUCCESS.value
                        if execution_result.success
                        else CompletionReason.FAILED.value
                    )
                )

            result = IntentResult(
                error=error,
                status=status,
                intent=intent,
                metrics=metrics,
                success=success,
                duration=duration,
                workflow_id=workflow_id,
                memory_summary=memory_summary,
                completion_reason=completion_reason,
                steps_taken=progress.get("step_count", 0),
                steps_executed=progress.get("step_count", 0),
                executed_subgoals=executed_subgoals,
                skipped_subgoals=skipped_subgoals,
                subgoal_count=subgoal_count,
                step_results=strategy.step_results,
            )

            terminal_event = self.__terminal_event(
                is_cancelled=is_cancelled,
                success=result.success,
            )
            terminal_message = self.__terminal_message(
                event=terminal_event,
                completion_reason=completion_reason,
            )

            logger.info(
                "Workflow outcome resolved",
                extra={
                    "event": "workflow.outcome.resolved",
                    "duration": duration,
                    "workflow.id": workflow_id,
                    "steps.taken": result.steps_taken,
                    "completion.reason": completion_reason,
                    "outcome": "cancelled"
                    if is_cancelled
                    else ("completed" if result.success else "failed"),
                },
            )

            await self.__telemetry.info(
                terminal_message,
                duration=duration,
                type=terminal_event,
                success=result.success,
                steps_taken=result.steps_taken,
                completion_reason=completion_reason,
            )

            return result

        finally:
            await self.__phase.shutdown()
            self.__current_strategy = None

    @staticmethod
    def __terminal_event(*, is_cancelled: bool, success: bool) -> FathomEvent:
        """
        Return the terminal workflow event matching the resolved outcome.
        """

        if is_cancelled:
            return FathomEvent.WORKFLOW_CANCELLED

        if success:
            return FathomEvent.WORKFLOW_COMPLETED

        return FathomEvent.WORKFLOW_FAILED

    @staticmethod
    def __terminal_message(*, event: FathomEvent, completion_reason: str) -> str:
        """
        Return the user-facing terminal workflow message.
        """

        if event is FathomEvent.WORKFLOW_CANCELLED:
            return "Run cancelled by operator."

        if event is FathomEvent.WORKFLOW_FAILED:
            reason = completion_reason.strip() or CompletionReason.FAILED.value
            return f"Run failed: {reason}"

        return "All wrapped up."

    async def run_exploration(
        self,
        max_steps: int = 100,
        request_id: Optional[str] = None,
        package_name: Optional[str] = None,
    ) -> ExplorationResult:
        """
        Execute exploration workflow.
        """

        start_time = time.time()
        workflow_id = request_id or uuid.uuid4().hex[:8]

        # Synchronize telemetry identity with workflow_id for routing
        # update_identity is available on some telemetry implementations
        telemetry_with_identity = cast("Any", self.__telemetry)
        if hasattr(telemetry_with_identity, "update_identity"):
            telemetry_with_identity.update_identity(identity=workflow_id)

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

        # Initialize context
        self.__context_manager = ContextManager(memory=self.__memory, workflow_id=workflow_id)
        self.__context_manager.set_roadmap(intent="Explore application structure")

        strategy = ExplorationStrategy(
            llm=self.__llm,
            device=self.__device,
            perception=self.__perception,
            memory=self.__memory,
            signal=self.__signal,
            storage=self.__storage,
            workflow_id=workflow_id,
            package_name=package_name,
            telemetry=self.__telemetry,
            configuration=self.__config,
            path_manager=self.__path_manager,
            seed=self.__config.exploration.random_seed,
            timeout=float(self.__config.exploration.timeout),
            max_steps=max_steps or self.__config.exploration.max_steps,
            runtime_configuration=self.__runtime_configuration,
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
            unique_screens = stats.get("unique_screens", 0)
            unexplored = stats.get("unexplored", 0)
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
                steps_executed=progress.get("steps", 0),
                coverage_percentage=coverage_percentage,
                completion_reason="Exploration completed",
                discovered_activities=discovered_activities,
                total_actions=stats.get("total_actions", 0),
                total_transitions=stats.get("total_transitions", 0),
                status="completed" if execution_result.success else "failed",
            )

            await self.__telemetry.info(
                "Exploration workflow completed",
                duration=duration,
                total_actions=result.total_actions,
                unique_screens=result.unique_screens,
            )

            return result

        finally:
            self.__current_strategy = None

    def cancel(self) -> None:
        """
        Cancel the currently running workflow.
        """

        if self.__current_strategy:
            logger.warning("Workflow cancellation requested")

            # Call cancel method on strategy if it has one
            strategy_with_cancel = cast("Any", self.__current_strategy)
            if hasattr(strategy_with_cancel, "cancel"):
                strategy_with_cancel.cancel()
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
        Cleanup all runner resources via abandonable phases so a stuck step cannot wedge the host.
        """

        cleanup_budget = self.__config.intent.finalization.runtime.cleanup

        async def __context_manager_shutdown() -> None:
            """
            Context manager shutdown wrapped for capture by AbandonablePhase.
            """

            if self.__context_manager is None:
                return
            try:
                await self.__context_manager.shutdown()
            except Exception as exception:
                logger.warning(
                    "context_manager shutdown failed: %s",
                    exception,
                    extra={
                        "event": "fathom.runner.cleanup.context_manager.failed",
                        "exception.type": type(exception).__name__,
                        "exception.message": str(exception),
                    },
                )

        async def __phase_shutdown() -> None:
            """
            Cancel any in-flight phase pulse so the heartbeat task cannot outlive the workflow.
            """

            try:
                await self.__phase.shutdown()
            except Exception as exception:
                logger.warning(
                    "phase announcer shutdown failed: %s",
                    exception,
                    extra={
                        "event": "fathom.runner.cleanup.phase.failed",
                        "exception.type": type(exception).__name__,
                        "exception.message": str(exception),
                    },
                )

        async def __llm_cleanup() -> None:
            """
            LLM port cleanup wrapped for capture by AbandonablePhase.
            """

            try:
                await self.__llm.cleanup()
            except Exception as exception:
                logger.warning(
                    "llm cleanup failed: %s",
                    exception,
                    extra={
                        "event": "fathom.runner.cleanup.llm.failed",
                        "exception.type": type(exception).__name__,
                        "exception.message": str(exception),
                    },
                )

        async def __owned_resources_cleanup() -> None:
            """
            Dedicated LLMs (e.g. qualifier LLM) handed over by the SDK builder path.

            Temporal / CLI callers pass nothing — they manage these via
            RunnerComposition.resources at the composition root; the list is
            empty for them.
            """

            for resource in self.__owned_resources:
                try:
                    await resource.cleanup()
                except Exception as exception:
                    logger.warning(
                        "owned resource cleanup failed: %s",
                        exception,
                        extra={
                            "event": "fathom.runner.cleanup.owned_resource.failed",
                            "exception.type": type(exception).__name__,
                            "exception.message": str(exception),
                        },
                    )

        async def __device_close() -> None:
            """
            Device port close wrapped for capture by AbandonablePhase.
            """

            if not hasattr(self.__device, "close"):
                return
            try:
                await self.__device.close()
            except Exception as exception:
                logger.warning(
                    "device close failed: %s",
                    exception,
                    extra={
                        "event": "fathom.runner.cleanup.device.failed",
                        "exception.type": type(exception).__name__,
                        "exception.message": str(exception),
                    },
                )

        async def __telemetry_close() -> None:
            """
            Telemetry port close wrapped for capture by AbandonablePhase.
            """

            if not hasattr(self.__telemetry, "close"):
                return
            try:
                await self.__telemetry.close()
            except Exception as exception:
                logger.warning(
                    "telemetry close failed: %s",
                    exception,
                    extra={
                        "event": "fathom.runner.cleanup.telemetry.failed",
                        "exception.type": type(exception).__name__,
                        "exception.message": str(exception),
                    },
                )

        async def __storage_close() -> None:
            """
            Storage port close wrapped for capture by AbandonablePhase.
            """

            if not hasattr(self.__storage, "close"):
                return
            try:
                await self.__storage.close()
            except Exception as exception:
                logger.warning(
                    "storage close failed: %s",
                    exception,
                    extra={
                        "event": "fathom.runner.cleanup.storage.failed",
                        "exception.type": type(exception).__name__,
                        "exception.message": str(exception),
                    },
                )

        for awaitable in (
            __phase_shutdown(),
            __context_manager_shutdown(),
            __llm_cleanup(),
            __owned_resources_cleanup(),
            __device_close(),
            __telemetry_close(),
            __storage_close(),
        ):
            await AbandonablePhase(
                phase=FinalizationPhase.RUNNER_CLEANUP,
                timeout=cleanup_budget,
            ).execute(awaitable=awaitable)

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

        A rejected intent must never touch the device — package lookup, ADB round-trips, anything.
        Returns a populated IntentResult when the gate blocks the run; returns None when the run is allowed to proceed.
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
            "Got it, getting started...", type=FathomEvent.INTENT_QUALIFIED, **verdict_payload
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
        Emit the rejection notification and produce a terminal IntentResult.

        The run is marked as completed because the intent has been fully handled by the
        qualifier; success is False because no UI action was attempted.
        """

        user_message = verdict.message or DEFAULT_REJECTION_MESSAGE
        duration = time.time() - start_time

        # INTENT_REJECTED carries the full verdict for clients that switch on it.
        await self.__telemetry.warning(
            user_message, type=FathomEvent.INTENT_REJECTED, **verdict_payload
        )

        logger.info(
            "[FathomRunner] Rejection emitted to client",
            extra={
                "duration": duration,
                "workflow_id": workflow_id,
                "completion_reason": CompletionReason.NOT_EXECUTABLE.value,
            },
        )

        # WORKFLOW_COMPLETED is dual-emitted so legacy consumers (Genymotion,
        # Temporal activity result handlers) that key off the terminal event
        # still receive a completion signal. The run *is* terminating here.
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
                "Runner: phase.shutdown after qualifier-reject failed: %s; terminal event already emitted",
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

    async def __get_memory_summary(self) -> Dict[str, Any]:
        """
        Get memory summary from memory port.
        """

        try:
            # Get all knowledge from memory provider
            knowledge = await self.__memory.get_all_knowledge()

            # Extract screens information
            screens = knowledge.get("screens", [])

            # Format for CLI display
            screens_formatted = []
            for screen in screens[:10]:  # Last 10 screens
                screens_formatted.append(
                    {
                        "hash": screen.get("hash", "")[:12],
                        "activity": screen.get("activity", "unknown"),
                        "description": screen.get("description", ""),
                    }
                )

            # Count experiences
            experience_count = knowledge.get("experience_count", 0)

            return {
                "screens": screens_formatted,
                "total_screens": len(screens),
                "experience_count": experience_count,
            }
        except Exception as exception:
            await self.__telemetry.warning(f"Failed to get memory summary: {exception}")
            return {
                "screens": [],
                "total_screens": 0,
                "experience_count": 0,
            }

    async def __export_graph(self, graph: ExplorationGraph) -> Dict[str, Any]:
        """
        Export exploration graph to dictionary.
        """

        try:
            nodes_dict = {}

            for fingerprint, node in graph.nodes.items():
                nodes_dict[fingerprint] = {
                    "visits": node.visits,
                    "activity": node.activity,
                    "actions": list(node.actions),
                    "transitions": node.transitions,
                }

            edges_list = [
                {"origin": origin, "action": action, "destination": dest}
                for origin, action, dest in graph.edges
            ]

            return {
                "nodes": nodes_dict,
                "edges": edges_list,
                "stats": graph.get_stats(),
            }
        except Exception as exception:
            await self.__telemetry.warning(f"Failed to export graph: {exception}")
            return {}
