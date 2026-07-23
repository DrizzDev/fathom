from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, Final, List, Optional, Tuple, cast

from fathom.constants import FathomEvent
from fathom.constants.collaboration import ArtifactBackend, TaskCode, TaskState
from fathom.constants.conversation import ProgressStatus, RecorderEvent
from fathom.constants.execution import LAUNCHER_PACKAGES
from fathom.constants.gcc import GCC_BRANCHING_ACTIVE_COUNT
from fathom.constants.messages import RECORDING_FAILURE_MESSAGE
from fathom.constants.state import (
    CommonStateKey,
    CompletionReason,
    IntentStateKey,
    VerifyMode,
)
from fathom.conversation.identity import InteractionIdentity
from fathom.core.exceptions import FathomError
from fathom.schemas.observation import ScreenObservation
from fathom.schemas.recording import (
    ActionSummary,
    Analysis,
    ContextSnapshot,
    Metrics,
    Observation,
    Output,
    StepCompletion,
    Usage,
)
from fathom.schemas.results import AnalysisResult, PlanResult
from fathom.schemas.screens import ScreenState
from fathom.schemas.steps import StepRecord, StepResult
from fathom.strategies.graph.intent.nodes.provider import IntentNodeProvider
from fathom.strategies.graph.intent.verification import VerificationModePolicy
from fathom.strategies.graph.state import IntentGraphState

logger = logging.getLogger(__name__)


class RecordNode:
    """
    RECORD graph node; commits results and decides task advancement.
    """

    __STEP_STARTED_AT: Final[str] = "started_at"

    def __init__(self, *, provider: IntentNodeProvider) -> None:
        """
        Bind the node to the shared intent provider.
        """

        self.__provider = provider
        self.__verification_modes = VerificationModePolicy()

    async def __call__(self, state: IntentGraphState) -> IntentGraphState:
        """
        Run the RECORD node handler.
        """

        return await self.run(state=state)

    async def run(self, *, state: IntentGraphState) -> IntentGraphState:
        """
        Record the execution result.

        ERROR BOUNDARY: Wraps recording in try/except to handle storage/telemetry failures gracefully.
        """

        record_start = time.time()

        logger.info(
            "Starting record node",
            extra={
                "component": "graph.intent.record",
                "event": "record.log",
                "workflow.id": self.__provider.context.workflow_id,
            },
        )

        if await self.__provider.is_cancelled():
            logger.warning(
                "Execution cancelled",
                extra={
                    "component": "graph.intent.record",
                    "event": "record.log",
                    "workflow.id": self.__provider.context.workflow_id,
                },
            )
            self.__provider.context.agent_state.mark_complete(
                reason=CompletionReason.CANCELLED.value
            )

            result = cast(
                "IntentGraphState",
                {
                    IntentStateKey.VERIFY_MODE: None,
                    CommonStateKey.IS_COMPLETE: True,
                    IntentStateKey.SHOULD_RETRY: False,
                    CommonStateKey.COMPLETION_REASON: CompletionReason.CANCELLED.value,
                },
            )
            self.__provider.persistence.persist(result=result)
            return result

        recorded_step = state.get(CommonStateKey.STEP_RESULT)
        if not isinstance(recorded_step, StepResult):
            message = (
                "Recording failed: missing StepResult; OBSERVE did not stage a recordable step."
            )
            logger.error(
                message,
                extra={
                    "component": "graph.intent.record",
                    "event": "record.missing.step_result",
                    "workflow.id": self.__provider.context.workflow_id,
                },
            )
            self.__provider.context.agent_state.mark_complete(reason=CompletionReason.FAILED.value)
            result = cast(
                "IntentGraphState",
                {
                    IntentStateKey.VERIFY_MODE: None,
                    IntentStateKey.SHOULD_RETRY: False,
                    CommonStateKey.IS_COMPLETE: True,
                    CommonStateKey.COMPLETION_REASON: CompletionReason.FAILED.value,
                    CommonStateKey.FAILURE_DIAGNOSTIC: message,
                },
            )
            self.__provider.persistence.persist(result=result)
            return result

        step_result: StepResult = recorded_step

        record: Optional[StepRecord] = None
        grounding_duration = analysis_duration = execution_duration = observe_duration = 0.0
        supervise_duration = 0.0
        step_started_at: Optional[float] = None

        # ERROR BOUNDARY: Wrap recording logic
        try:
            # Record in agent state (internal bookkeeping, always done)
            self.__provider.context.agent_state.record_step(result=step_result)
            await self.__record_step_finished(result=step_result, state=state)

            # Accumulate step results in graph state so MemorySaver checkpoints them.
            existing_step_results = cast(
                "List[StepResult]", state.get(IntentStateKey.STEP_RESULTS) or []
            )
            accumulated_step_results = existing_step_results + [step_result]

            # Use post-action activity captured in EXECUTE node (avoids extra device call)
            post_activity_raw = state.get(IntentStateKey.POST_ACTIVITY)
            current_activity = str(post_activity_raw) if post_activity_raw else "unknown"
            execution_activity = "unknown"
            screen_state_value = state.get(CommonStateKey.SCREEN_STATE)
            if isinstance(screen_state_value, ScreenState):
                execution_activity = screen_state_value.activity or "unknown"

            logger.info(
                f"Recording step: success={step_result.success}, "
                f"screen_changed={step_result.screen_changed}, duration={step_result.duration}ms, "
                f"execution_package={execution_activity}, observed_package={current_activity}"
            )

            # LAUNCHER BLOCKING: Never persist actions taken on launcher apps
            execution_package_base = execution_activity.split("/")[0]
            observed_package_base = current_activity.split("/")[0]
            is_on_launcher = self.__provider.persistence.should_skip_launcher(
                execution_activity=execution_activity,
                observed_activity=current_activity,
            )

            if is_on_launcher:
                logger.warning(
                    f"Skipping persistence: on launcher app. "
                    f"Launcher={execution_package_base}, "
                    f"Observed={observed_package_base}, "
                    f"step_num={step_result.step.step_number}, action_type={step_result.step.action.action_type.value}"
                )
                await self.__provider.context.telemetry.warning(
                    f"Step {step_result.step.step_number + 1} not persisted (on launcher)",
                    execution_package=execution_activity,
                    observed_package=current_activity,
                    step_number=step_result.step.step_number + 1,
                    action_type=step_result.step.action.action_type.value,
                )
            else:
                self.__provider.persistence.enqueue_history(
                    step_result=step_result,
                    current_activity=current_activity,
                    execution_activity=execution_activity,
                )
                logger.info(
                    f"Recording step to history. Observed={current_activity}",
                    extra={
                        "component": "graph.intent.record",
                        "event": "record.log",
                        "workflow.id": self.__provider.context.workflow_id,
                    },
                )

            # Emit enriched telemetry for the UI to render full step details
            record = step_result.to_record(activity=current_activity)

            analysis_duration_raw = state.get(CommonStateKey.ANALYSIS_DURATION) or 0.0
            grounding_duration_raw = state.get(CommonStateKey.GROUNDING_DURATION) or 0.0
            execution_duration_raw = state.get(CommonStateKey.EXECUTION_DURATION) or 0.0

            analysis_duration = (
                float(analysis_duration_raw)
                if isinstance(analysis_duration_raw, (int, float, str))
                else 0.0
            )
            grounding_duration = (
                float(grounding_duration_raw)
                if isinstance(grounding_duration_raw, (int, float, str))
                else 0.0
            )
            execution_duration = (
                float(execution_duration_raw)
                if isinstance(execution_duration_raw, (int, float, str))
                else 0.0
            )
            observe_duration_raw = state.get(CommonStateKey.OBSERVE_DURATION) or 0.0
            observe_duration = (
                float(observe_duration_raw)
                if isinstance(observe_duration_raw, (int, float, str))
                else 0.0
            )
            supervise_duration_raw = state.get(CommonStateKey.SUPERVISE_DURATION) or 0.0
            supervise_duration = (
                float(supervise_duration_raw)
                if isinstance(supervise_duration_raw, (int, float, str))
                else 0.0
            )

            step_started_at_raw = state.get(CommonStateKey.STEP_STARTED_AT)
            step_started_at = (
                float(step_started_at_raw)
                if isinstance(step_started_at_raw, (int, float))
                else None
            )

            # Console audit logging for rich step visualization
            current_screen = state.get(CommonStateKey.SCREEN_STATE)
            is_new_screen_state = state.get(CommonStateKey.IS_NEW_SCREEN)
            execution_plan = state.get(IntentStateKey.PLAN)

            if (
                isinstance(execution_plan, PlanResult)
                and isinstance(current_screen, ScreenState)
                and isinstance(is_new_screen_state, bool)
            ):
                self.__provider.context.auditor.log_step(
                    plan=execution_plan,
                    state=current_screen,
                    hierarchy_duration=0.0,
                    is_new_screen=is_new_screen_state,
                    result=step_result.to_record(),
                    analysis_duration=analysis_duration,
                    execution_duration=execution_duration,
                    grounding_duration=grounding_duration,
                    is_stuck=self.__provider.context.agent_state.is_stuck,
                    step_count=self.__provider.context.agent_state.step_count,
                    total_duration=grounding_duration + analysis_duration + execution_duration,
                )

            # SCRIPT_GENERATED is emitted only when the run completes (intent strategy),
            # not on every step, to avoid sending stale script content to the client.

            if execution_package_base not in LAUNCHER_PACKAGES:
                await self.__provider.context.memory.store_experience(
                    success=step_result.success,
                    action=step_result.step.action,
                    visual_hash=step_result.pre_hash,
                )

            logger.info(
                f"[H3] Committing to trace | thought={step_result.step.action.rationale[:50]}..."
            )

            analysis_result = state.get(CommonStateKey.ANALYSIS)
            analysis: Optional[AnalysisResult] = None
            if isinstance(analysis_result, AnalysisResult):
                analysis = analysis_result

            observation = f"Screen: {step_result.pre_hash[:8]}"
            if analysis and analysis.screen_description:
                observation += f" | Content: {analysis.screen_description[:100]}..."

            await self.__provider.context.context_manager.commit(
                observation=observation,
                action=step_result.step.action,
                thought=step_result.step.action.rationale,
            )

            full_context = self.__provider.context.context_manager.get_full_context()
            active_count = full_context.get("active_count", 0)

            if active_count >= GCC_BRANCHING_ACTIVE_COUNT:
                logger.info(
                    "Record node triggering GCC branch",
                    extra={
                        "component": "graph.intent.record",
                        "event": "record.gcc.branch",
                        "active.count": active_count,
                        "workflow.id": self.__provider.context.workflow_id,
                    },
                )
                await self.__provider.context.context_manager.branch()

            execution_plan = state.get(IntentStateKey.PLAN)

            if isinstance(execution_plan, PlanResult) and execution_plan.is_complete:
                verify_mode = self.__verification_modes.mode_for_verify(
                    state=state,
                    agent_state=self.__provider.context.agent_state,
                )
                logger.info(
                    "Plan indicates completion. This is the final step.",
                    extra={
                        "component": "graph.intent.record",
                        "event": "record.log",
                        "verify.mode": verify_mode.value,
                        "workflow.id": self.__provider.context.workflow_id,
                    },
                )

                self.__provider.context.agent_state.clear_verification_loop()
                self.__provider.context.agent_state.reset_complete_deferrals()
                completion_reason = self.__completion_claim_reason(reason=execution_plan.reason)
                if verify_mode is VerifyMode.FULL_INTENT:
                    self.__provider.context.agent_state.mark_complete(reason=completion_reason)

                result = cast(
                    "IntentGraphState",
                    {
                        CommonStateKey.IS_COMPLETE: True,
                        IntentStateKey.SHOULD_RETRY: False,
                        IntentStateKey.VERIFY_MODE: verify_mode.value,
                        IntentStateKey.STEP_RESULTS: accumulated_step_results,
                        CommonStateKey.COMPLETION_REASON: completion_reason,
                    },
                )
                await self.__emit_step_completed(
                    state=state,
                    step_result=step_result,
                    record=record,
                    grounding_duration=grounding_duration,
                    analysis_duration=analysis_duration,
                    execution_duration=execution_duration,
                    observe_duration=observe_duration,
                    supervise_duration=supervise_duration,
                    record_start=record_start,
                    step_started_at=step_started_at,
                )
                self.__provider.persistence.persist(result=result)
                return result

            # ── Sub-goal completion check (post-execution) ──
            # Evaluated here — after the action has executed and been recorded —
            # so we never advance a sub-goal on an action that didn't run.
            # The typed ScreenObservation (post-action capture written by the
            # OBSERVE node) is the source of truth for criterion satisfaction;
            # we pass it explicitly so the gate never falls back to the
            # free-text observation field on StepResult.
            screen_observation_raw = state.get(CommonStateKey.SCREEN_OBSERVATION)
            screen_observation = (
                screen_observation_raw
                if isinstance(screen_observation_raw, ScreenObservation)
                else None
            )
            subgoal_result = await self.__provider.completion.evaluate(
                plan=execution_plan,
                step_result=step_result,
                accumulated=accumulated_step_results,
                observation=screen_observation,
            )
            if subgoal_result is not None:
                await self.__emit_step_completed(
                    state=state,
                    step_result=step_result,
                    record=record,
                    grounding_duration=grounding_duration,
                    analysis_duration=analysis_duration,
                    execution_duration=execution_duration,
                    observe_duration=observe_duration,
                    supervise_duration=supervise_duration,
                    record_start=record_start,
                    step_started_at=step_started_at,
                )
                self.__provider.persistence.persist(result=subgoal_result)
                return subgoal_result

            logger.info(
                f"Step {self.__provider.context.agent_state.step_count} recorded successfully",
                extra={
                    "component": "graph.intent.record",
                    "event": "record.log",
                    "workflow.id": self.__provider.context.workflow_id,
                },
            )
            logger.info(
                "-> Will route to GROUND for next step",
                extra={
                    "component": "graph.intent.record",
                    "event": "record.log",
                    "workflow.id": self.__provider.context.workflow_id,
                },
            )

            result = cast(
                "IntentGraphState",
                {
                    IntentStateKey.VERIFY_MODE: None,
                    IntentStateKey.STEP_RESULTS: accumulated_step_results,
                },
            )
            await self.__emit_step_completed(
                state=state,
                step_result=step_result,
                record=record,
                grounding_duration=grounding_duration,
                analysis_duration=analysis_duration,
                execution_duration=execution_duration,
                observe_duration=observe_duration,
                supervise_duration=supervise_duration,
                record_start=record_start,
                step_started_at=step_started_at,
            )
            self.__provider.persistence.persist(result=result)
            return result

        except asyncio.CancelledError:
            # CancelledError is the cooperative cancellation signal; it
            # must propagate so the LangGraph task tree unwinds cleanly.
            raise
        except Exception as exception:
            logger.exception(
                f"Recording failed: {exception}",
                extra={
                    "component": "graph.intent.record",
                    "event": "record.log",
                    "workflow.id": self.__provider.context.workflow_id,
                },
            )
            display_error = (
                exception.display(fallback=RECORDING_FAILURE_MESSAGE)
                if isinstance(exception, FathomError)
                else RECORDING_FAILURE_MESSAGE
            )
            await self.__provider.context.telemetry.error(
                display_error,
                step=self.__provider.context.agent_state.step_count,
            )
            if record is not None:
                try:
                    await self.__emit_step_completed(
                        state=state,
                        step_result=step_result,
                        record=record,
                        grounding_duration=grounding_duration,
                        analysis_duration=analysis_duration,
                        execution_duration=execution_duration,
                        observe_duration=observe_duration,
                        supervise_duration=supervise_duration,
                        record_start=record_start,
                        step_started_at=step_started_at,
                    )
                except Exception:
                    logger.exception("Failed to emit STEP_COMPLETED after recording failure")
            existing_step_results = cast(
                "List[StepResult]", state.get(IntentStateKey.STEP_RESULTS) or []
            )
            self.__provider.context.agent_state.mark_complete(reason=CompletionReason.FAILED.value)
            self.__provider.context.agent_state.clear_verification_loop()
            self.__provider.context.agent_state.reset_complete_deferrals()
            result = cast(
                "IntentGraphState",
                {
                    IntentStateKey.VERIFY_MODE: None,
                    IntentStateKey.SHOULD_RETRY: False,
                    IntentStateKey.STEP_RESULTS: existing_step_results,
                    CommonStateKey.IS_COMPLETE: True,
                    CommonStateKey.FAILURE_DIAGNOSTIC: display_error,
                    CommonStateKey.COMPLETION_REASON: CompletionReason.FAILED.value,
                },
            )
            self.__provider.persistence.persist(result=result)
            return result

    async def __emit_step_completed(
        self,
        *,
        state: IntentGraphState,
        step_result: StepResult,
        record: StepRecord,
        grounding_duration: float,
        analysis_duration: float,
        execution_duration: float,
        observe_duration: float,
        supervise_duration: float,
        record_start: float,
        step_started_at: Optional[float],
    ) -> None:
        """
        Emit STEP_COMPLETED telemetry with the step's final timing breakdown.
        """

        plan_metrics: Dict[str, Any] = {}
        plan_raw = state.get(IntentStateKey.PLAN)
        if isinstance(plan_raw, PlanResult):
            plan_metrics = dict(plan_raw.metrics or {})

        artifacts_payload = (
            step_result.artifacts.model_dump(mode="json")
            if step_result.artifacts is not None
            else None
        )

        record_duration_ms = int((time.time() - record_start) * 1000)

        if step_started_at is not None:
            total_duration = int((time.time() - step_started_at) * 1000)
        else:
            total_duration = (
                int((grounding_duration + analysis_duration + execution_duration) * 1000)
                + record_duration_ms
            )

        await self.__provider.context.telemetry.info(
            f"Step {step_result.step.step_number + 1} completed",
            type=FathomEvent.STEP_COMPLETED,
            success=record.success,
            duration=total_duration,
            grounding_ms=int(grounding_duration * 1000),
            analysis_ms=int(analysis_duration * 1000),
            execution_ms=int(execution_duration * 1000),
            observe_ms=int(observe_duration * 1000),
            supervise_ms=int(supervise_duration * 1000),
            record_ms=record_duration_ms,
            artifacts=artifacts_payload,
            rationale=record.rationale,
            observation=record.observation,
            action_type=record.action_type,
            step=step_result.step.step_number + 1,
            action_description=record.action_description,
            target=record.natural_language_target or record.target,
            analysis_llm_ms=float(plan_metrics.get("llm_analysis_ms", 0.0) or 0.0),
            analysis_parse_ms=float(plan_metrics.get("parse_ms", 0.0) or 0.0),
            analysis_payload_ms=float(plan_metrics.get("payload_ms", 0.0) or 0.0),
            analysis_manifest_ms=float(plan_metrics.get("manifest_ms", 0.0) or 0.0),
            analysis_tool_scope_ms=float(plan_metrics.get("tool_scope_ms", 0.0) or 0.0),
            analysis_total_ms=float(plan_metrics.get("analyze_ms", 0.0) or 0.0),
        )

    async def __record_step_finished(self, *, result: StepResult, state: IntentGraphState) -> None:
        """
        Record progress, artifacts, context, and terminal state for one graph action task.
        """

        context = self.__provider.context
        recorder = getattr(context, "recorder", None)
        tenant = getattr(context, "tenant", None)
        thread = getattr(context, "thread", None)
        responder = getattr(context, "responder", None)
        workspace = getattr(context, "workspace", None)
        execution_id = getattr(context, "execution_id", None)
        workflow_id = getattr(context, "workflow_id", None)
        if recorder is None:
            return
        if (
            not isinstance(tenant, str)
            or not isinstance(thread, str)
            or not isinstance(responder, str)
            or not isinstance(execution_id, str)
            or not isinstance(workflow_id, str)
        ):
            return
        if workspace is not None and not isinstance(workspace, str):
            return

        identity = InteractionIdentity(execution=execution_id)

        step_task_id = identity.step_task(
            step_number=result.step.step_number,
            action_descriptor=result.step.action.to_description(),
        )
        progress = await self.__record_progress_message(
            state=state,
            result=result,
            step_task_id=step_task_id,
        )
        artifacts = await self.__record_step_artifacts(
            step_task_id=step_task_id,
            step_number=result.step.step_number,
        )
        await self.__record_step_context(
            result=result,
            progress=progress,
            artifacts=artifacts,
            step_task_id=step_task_id,
        )

        try:
            await recorder.record_step_finished(
                completion=StepCompletion(
                    tenant=tenant,
                    thread=thread,
                    task=step_task_id,
                    reason=result.error,
                    workflow=workflow_id,
                    elapsed=result.duration,
                    finished=datetime.now(tz=timezone.utc),
                    summary=result.observation or result.step.action.to_description(),
                    state=TaskState.SUCCEEDED if result.success else TaskState.FAILED,
                    code=TaskCode.COMPLETED if result.success else TaskCode.UNKNOWN_ERROR,
                )
            )
        except Exception as exception:
            await context.telemetry.warning(
                "Conversation step finish recording failed",
                error=str(exception),
                step=result.step.step_number + 1,
            )
            return

    async def __record_progress_message(
        self,
        *,
        step_task_id: str,
        result: StepResult,
        state: IntentGraphState,
    ) -> Optional[str]:
        """
        Record one user-visible progress message for the completed graph step.
        """

        context = self.__provider.context
        recorder = getattr(context, "recorder", None)
        tenant = getattr(context, "tenant", None)
        thread = getattr(context, "thread", None)
        responder = getattr(context, "responder", None)
        workspace = getattr(context, "workspace", None)
        execution_id = getattr(context, "execution_id", None)
        workflow_id = getattr(context, "workflow_id", None)

        if recorder is None:
            return None

        if (
            not isinstance(tenant, str)
            or not isinstance(thread, str)
            or not isinstance(responder, str)
            or not isinstance(execution_id, str)
            or not isinstance(workflow_id, str)
        ):
            return None
        if workspace is not None and not isinstance(workspace, str):
            return None

        analysis_raw = state.get(CommonStateKey.ANALYSIS)
        analysis = analysis_raw if isinstance(analysis_raw, AnalysisResult) else None

        action = result.step.action
        identity = InteractionIdentity(execution=execution_id)
        message = identity.derived_message(
            name="progress",
            qualifier=f"{result.step.step_number}:{action.to_description()}",
        )
        metrics = self.__progress_metrics(result=result, state=state, analysis=analysis)

        try:
            await recorder.record_llm_analysis(
                analysis=Analysis(
                    id=message,
                    thread=thread,
                    tenant=tenant,
                    metrics=metrics,
                    actor=responder,
                    task=step_task_id,
                    workspace=workspace,
                    workflow=workflow_id,
                    execution=execution_id,
                    step=result.step.step_number + 1,
                    status=(
                        ProgressStatus.COMPLETED.value
                        if result.success
                        else ProgressStatus.FAILED.value
                    ),
                    rationale=action.rationale,
                    summary=self.__progress_summary(result=result, analysis=analysis),
                    evidence=self.__progress_evidence(result=result, analysis=analysis),
                    action=self.__progress_action(result=result),
                    created=self.__progress_created(result=result),
                    observation=self.__progress_observation(result=result, analysis=analysis),
                    metadata={
                        "screen_hash": result.step.screen_hash,
                        "metrics": metrics.model_dump(mode="json", exclude_none=True),
                        "analysis_present": analysis is not None,
                    },
                )
            )
            await context.telemetry.info(
                "Conversation progress message recorded.",
                type=RecorderEvent.TIMELINE_PROGRESS_RECORDED.value,
                tenant_id=tenant,
                conversation_id=thread,
                execution_id=getattr(context, "execution_id", None),
                workflow_id=workflow_id,
                task=step_task_id,
                step=result.step.step_number + 1,
                duration=self.__duration_seconds(metrics=metrics),
                analysis_present=analysis is not None,
                step_result_present=True,
            )
            return message
        except Exception as exception:
            await context.telemetry.warning(
                "Conversation progress message recording failed",
                type=RecorderEvent.TIMELINE_PROGRESS_FAILED.value,
                tenant_id=tenant,
                conversation_id=thread,
                execution_id=getattr(context, "execution_id", None),
                workflow_id=workflow_id,
                task=step_task_id,
                error=str(exception),
                step=result.step.step_number + 1,
                duration=self.__duration_seconds(metrics=metrics),
                analysis_present=analysis is not None,
                step_result_present=True,
            )
            return None

    def __progress_action(self, *, result: StepResult) -> ActionSummary:
        """
        Build the user-safe action projection for a progress message.
        """

        action = result.step.action

        return ActionSummary(
            text=action.text,
            rationale=action.rationale,
            confidence=action.confidence,
            type=action.action_type.value,
            target=action.natural_language_target or action.target,
        )

    def __progress_observation(
        self, *, result: StepResult, analysis: Optional[AnalysisResult]
    ) -> Observation:
        """
        Build the user-safe observation projection for a progress message.
        """

        return Observation(
            summary=result.observation,
            changed=result.screen_changed,
            screen=result.step.screen_hash,
            evidence=self.__progress_evidence(result=result, analysis=analysis),
        )

    @staticmethod
    def __progress_summary(*, result: StepResult, analysis: Optional[AnalysisResult]) -> str:
        """
        Return the best available user-safe step summary.
        """

        if analysis is not None and analysis.reasoning:
            return analysis.reasoning

        if result.observation:
            return result.observation

        return result.step.action.to_description()

    @staticmethod
    def __progress_evidence(
        *, result: StepResult, analysis: Optional[AnalysisResult]
    ) -> Optional[str]:
        """
        Return screen evidence from analysis or the post-action observation.
        """

        if analysis is not None and analysis.screen_description:
            return analysis.screen_description

        return result.observation

    def __progress_metrics(
        self,
        *,
        result: StepResult,
        state: IntentGraphState,
        analysis: Optional[AnalysisResult],
    ) -> Metrics:
        """
        Build timing and token metadata for a progress message.
        """

        analysis_duration = self.__duration(value=state.get(CommonStateKey.ANALYSIS_DURATION))
        grounding_duration = self.__duration(value=state.get(CommonStateKey.GROUNDING_DURATION))
        execution_duration = self.__duration(value=state.get(CommonStateKey.EXECUTION_DURATION))

        return Metrics(
            analysis=analysis_duration,
            grounding=grounding_duration,
            execution=execution_duration or result.duration,
            total=self.__total_duration(
                values=(
                    analysis_duration,
                    grounding_duration,
                    execution_duration or result.duration,
                )
            ),
            usage=self.__usage(metrics=analysis.metrics if analysis is not None else {}),
        )

    @staticmethod
    def __duration(*, value: object) -> Optional[int]:
        """
        Convert optional second-based state timing into milliseconds.
        """

        if not isinstance(value, (float, int)):
            return None

        return max(0, int(float(value) * 1000))

    @staticmethod
    def __duration_seconds(*, metrics: Metrics) -> Optional[float]:
        """
        Return the total progress duration in seconds for observability.
        """

        if metrics.total is None:
            return None

        return metrics.total / 1000

    @staticmethod
    def __total_duration(*, values: Tuple[Optional[int], ...]) -> Optional[int]:
        """
        Sum available duration values.
        """

        available = tuple(value for value in values if value is not None)
        if not available:
            return None

        return sum(available)

    @staticmethod
    def __usage(*, metrics: Dict[str, float]) -> Optional[Usage]:
        """
        Convert provider token metrics into recorder usage metadata.
        """

        if not metrics:
            return None

        prompt = int(metrics.get("prompt_tokens", 0) or 0)
        cached = int(metrics.get("cached_tokens", 0) or 0)
        reasoning = int(metrics.get("reasoning_tokens", 0) or 0)
        completion = int(metrics.get("completion_tokens", 0) or 0)
        total = int(metrics.get("total_tokens", 0) or prompt + completion)

        if not any((prompt, completion, cached, reasoning, total)):
            return None

        return Usage(
            total=total or None,
            prompt=prompt or None,
            cached=cached or None,
            reasoning=reasoning or None,
            completion=completion or None,
        )

    def __progress_created(self, *, result: StepResult) -> datetime:
        """
        Return the deterministic creation timestamp carried by EXECUTE.
        """

        value = result.step.metadata.get(self.__STEP_STARTED_AT)

        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value)
            except ValueError:
                pass

        return datetime.now(tz=timezone.utc)

    async def __record_step_context(
        self,
        *,
        step_task_id: str,
        result: StepResult,
        progress: Optional[str],
        artifacts: Tuple[str, ...],
    ) -> None:
        """
        Record one audit context snapshot for a completed graph step.
        """

        context = self.__provider.context

        recorder = getattr(context, "recorder", None)

        tenant = getattr(context, "tenant", None)
        thread = getattr(context, "thread", None)
        responder = getattr(context, "responder", None)
        workspace = getattr(context, "workspace", None)
        execution_id = getattr(context, "execution_id", None)
        workflow_id = getattr(context, "workflow_id", None)

        if recorder is None:
            return
        if (
            not isinstance(tenant, str)
            or not isinstance(thread, str)
            or not isinstance(responder, str)
            or not isinstance(execution_id, str)
            or not isinstance(workflow_id, str)
        ):
            return
        if workspace is not None and not isinstance(workspace, str):
            return

        identity = InteractionIdentity(execution=execution_id)
        messages = (progress,) if progress is not None else ()
        try:
            await recorder.record_context(
                snapshot=ContextSnapshot(
                    tenant=tenant,
                    thread=thread,
                    actor=responder,
                    task=step_task_id,
                    messages=messages,
                    artifacts=artifacts,
                    workspace=workspace,
                    workflow=workflow_id,
                    execution=identity.execution,
                    created=self.__progress_created(result=result),
                    id=identity.context(name=f"step:{result.step.step_number}"),
                    hash=self.__context_hash(
                        messages=messages,
                        artifacts=artifacts,
                        workflow=workflow_id,
                        step=result.step.step_number,
                    ),
                    metadata={
                        "step": result.step.step_number,
                    },
                )
            )
        except Exception as exception:
            await context.telemetry.warning(
                "Conversation step context recording failed",
                error=str(exception),
                step=result.step.step_number + 1,
            )

    @staticmethod
    def __context_hash(
        *,
        step: int,
        workflow: str,
        messages: Tuple[str, ...],
        artifacts: Tuple[str, ...],
    ) -> str:
        """
        Return a stable digest for one step context snapshot.
        """

        material = "\x1f".join((workflow, str(step), *messages, *artifacts))
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    async def __record_step_artifacts(
        self, *, step_number: int, step_task_id: str
    ) -> Tuple[str, ...]:
        """
        Record artifacts emitted during one graph step.
        """

        context = self.__provider.context

        recorder = getattr(context, "recorder", None)
        workflow_id = getattr(context, "workflow_id", None)
        package_name = getattr(context, "package_name", None)

        if recorder is None:
            return ()

        if not isinstance(workflow_id, str) or not isinstance(package_name, str):
            return ()

        catalog = context.artifact_catalog
        try:
            logger.info(
                "Recording step artifacts",
                extra={
                    "event": "conversation.artifacts.step.started",
                    "step.index": step_number,
                    "workflow.id": workflow_id,
                },
            )
            artifacts = await catalog.discover(
                workflow=workflow_id,
                only_step=step_number,
                package_name=package_name,
            )
        except Exception as exception:
            await context.telemetry.warning(
                "Step artifact discovery failed",
                error=str(exception),
                step=step_number + 1,
            )
            return ()

        recorded: List[str] = []
        execution_id = context.execution_id
        identity = InteractionIdentity(execution=execution_id)

        for path, stat in artifacts:
            captured_at = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
            resolved_kind = catalog.kind(path=path)

            if resolved_kind is None:
                logger.warning(
                    "Skipping unclassified step artifact",
                    extra={
                        "event": "conversation.artifacts.step.skipped",
                        "path": str(path),
                        "step.index": step_number,
                        "workflow.id": workflow_id,
                    },
                )
                continue

            artifact = identity.artifact(path=path)

            metadata: Dict[str, Any] = {
                "step": step_number,
                "filename": path.name,
                "category": catalog.category(path=path),
                "captured_at": captured_at.isoformat(),
            }
            try:
                await recorder.record_artifact(
                    output=Output(
                        id=artifact,
                        uri=str(path),
                        task=step_task_id,
                        size=stat.st_size,
                        metadata=metadata,
                        kind=resolved_kind,
                        created=captured_at,
                        thread=context.thread,
                        tenant=context.tenant,
                        actor=context.responder,
                        workspace=context.workspace,
                        workflow=context.workflow_id,
                        execution=execution_id,
                        mime=catalog.mime(path=path),
                        backend=ArtifactBackend.LOCAL,  # TODO: Remove hardcoded value
                        retention=catalog.retention(path=path),
                    )
                )
                recorded.append(artifact)
            except Exception as exception:
                await context.telemetry.warning(
                    "Step artifact recording failed",
                    step=step_number + 1,
                    artifact=str(path),
                    error=str(exception),
                )
        logger.info(
            "Recorded step artifacts",
            extra={
                "event": "conversation.artifacts.step.completed",
                "step.index": step_number,
                "workflow.id": workflow_id,
                "artifacts.count": len(recorded),
            },
        )
        return tuple(recorded)

    @staticmethod
    def __completion_claim_reason(*, reason: Optional[str]) -> str:
        """
        Return a non-empty completion claim reason for graph state and AgentState.
        """

        return (reason or "").strip() or CompletionReason.SUCCESS.value
