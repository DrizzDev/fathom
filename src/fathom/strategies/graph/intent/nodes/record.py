from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional, cast

from fathom.constants import FathomEvent
from fathom.constants.execution import LAUNCHER_PACKAGES
from fathom.constants.gcc import GCC_BRANCHING_ACTIVE_COUNT
from fathom.constants.messages import RECORDING_FAILURE_MESSAGE
from fathom.constants.state import CommonStateKey, CompletionReason, IntentStateKey
from fathom.core.exceptions import FathomError
from fathom.schemas.observation import ScreenObservation
from fathom.schemas.results import AnalysisResult, PlanResult
from fathom.schemas.screens import ScreenState
from fathom.schemas.steps import StepResult
from fathom.strategies.graph.intent.nodes.provider import IntentNodeProvider
from fathom.strategies.graph.state import IntentGraphState

logger = logging.getLogger(__name__)


class RecordNode:
    """
    RECORD graph node; commits results and decides task advancement.
    """

    def __init__(self, *, provider: IntentNodeProvider) -> None:
        """
        Bind the node to the shared intent provider.
        """

        self.__provider = provider

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

            return cast(
                "IntentGraphState",
                {
                    CommonStateKey.IS_COMPLETE: True,
                    CommonStateKey.COMPLETION_REASON: CompletionReason.CANCELLED.value,
                },
            )

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
                    IntentStateKey.SHOULD_RETRY: False,
                    CommonStateKey.IS_COMPLETE: True,
                    CommonStateKey.COMPLETION_REASON: CompletionReason.FAILED.value,
                    CommonStateKey.FAILURE_DIAGNOSTIC: message,
                },
            )
            self.__provider.persistence.persist(result=result)
            return result

        step_result: StepResult = recorded_step

        # ERROR BOUNDARY: Wrap recording logic
        try:
            # Record in agent state (internal bookkeeping, always done)
            self.__provider.context.agent_state.record_step(result=step_result)

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

            total_duration = int(
                (grounding_duration + analysis_duration + execution_duration) * 1000
            )
            plan_metrics: Dict[str, Any] = {}
            plan_raw = state.get(IntentStateKey.PLAN)
            if isinstance(plan_raw, PlanResult):
                plan_metrics = dict(plan_raw.metrics or {})

            artifacts_payload = (
                step_result.artifacts.model_dump(mode="json")
                if step_result.artifacts is not None
                else None
            )

            await self.__provider.context.telemetry.info(
                f"Step {step_result.step.step_number + 1} completed",
                type=FathomEvent.STEP_COMPLETED,
                success=record.success,
                duration=total_duration,
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
                        "workflow.id": self.__provider.context.workflow_id,
                        "active.count": active_count,
                    },
                )
                await self.__provider.context.context_manager.branch()

            execution_plan = state.get(IntentStateKey.PLAN)

            if isinstance(execution_plan, PlanResult) and execution_plan.is_complete:
                logger.info(
                    "Plan indicates completion. This is the final step.",
                    extra={
                        "component": "graph.intent.record",
                        "event": "record.log",
                        "workflow.id": self.__provider.context.workflow_id,
                    },
                )
                self.__provider.context.agent_state.mark_complete(
                    reason=execution_plan.reason or "Completed"
                )

                result = cast(
                    "IntentGraphState",
                    {
                        CommonStateKey.IS_COMPLETE: True,
                        CommonStateKey.COMPLETION_REASON: execution_plan.reason,
                        IntentStateKey.STEP_RESULTS: accumulated_step_results,
                    },
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
                {IntentStateKey.STEP_RESULTS: accumulated_step_results},
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
            existing_step_results = cast(
                "List[StepResult]", state.get(IntentStateKey.STEP_RESULTS) or []
            )
            result = cast("IntentGraphState", {IntentStateKey.STEP_RESULTS: existing_step_results})
            self.__provider.persistence.persist(result=result)
            return result
