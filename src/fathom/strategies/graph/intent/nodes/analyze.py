from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, Optional, cast

from fathom.constants import FathomEvent
from fathom.constants.retries import (
    PLANNER_RETRY_CONSUMED,
    PLANNER_RETRY_EXHAUSTED,
    PLANNER_RETRY_METADATA_INVALID,
    RetryBranch,
    RetryKind,
    RetryMetadataField,
)
from fathom.constants.runtime import DEFAULT_COMPLETE_DEFERRAL_BUDGET
from fathom.constants.state import (
    TERMINAL_COMPLETION_REASONS,
    CommonStateKey,
    CompletionReason,
    IntentStateKey,
)
from fathom.schemas.observation import ScreenObservation
from fathom.schemas.screens import ScreenCapture
from fathom.strategies.graph.intent.nodes.provider import IntentNodeProvider
from fathom.strategies.graph.intent.verification import VerificationModePolicy
from fathom.strategies.graph.state import IntentGraphState

logger = logging.getLogger(__name__)


class AnalyzeNode:
    """
    ANALYZE graph node; plans the next step from the observation.
    """

    def __init__(self, *, provider: IntentNodeProvider) -> None:
        """
        Bind the node to the shared intent provider.
        """

        self.__provider = provider
        self.__verification_modes = VerificationModePolicy()

    async def __call__(self, state: IntentGraphState) -> IntentGraphState:
        """
        Run the ANALYZE node handler.
        """

        return await self.run(state=state)

    async def run(self, *, state: IntentGraphState) -> IntentGraphState:
        """
        Plan the next step using the Agent Planner.

        ERROR BOUNDARY: Wraps planning in try/except to handle LLM/network failures gracefully.
        """

        logger.info(
            "Starting analysis node",
            extra={
                "event": "analyze.log",
                "component": "graph.intent.analyze",
                "workflow.id": self.__provider.context.workflow_id,
            },
        )

        # Restore agent_state from graph checkpoint if available
        self.__provider.persistence.restore(state=state)

        if await self.__provider.is_cancelled():
            logger.warning(
                "Execution cancelled",
                extra={
                    "event": "analyze.log",
                    "component": "graph.intent.analyze",
                    "workflow.id": self.__provider.context.workflow_id,
                },
            )
            self.__provider.context.agent_state.mark_complete(
                reason=CompletionReason.CANCELLED.value
            )
            self.__provider.context.agent_state.reset_complete_deferrals()

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

        if self.__provider.context.agent_state.step_count >= self.__provider.context.max_steps:
            logger.warning(
                "Analysis reached max steps before planning; terminating workflow",
                extra={
                    "component": "graph.intent.analyze",
                    "event": "analyze.max.steps.terminated",
                    "max_steps": self.__provider.context.max_steps,
                    "workflow.id": self.__provider.context.workflow_id,
                    "step_count": self.__provider.context.agent_state.step_count,
                },
            )
            self.__provider.context.agent_state.mark_complete(
                reason=CompletionReason.MAX_STEPS.value
            )
            self.__provider.context.agent_state.reset_complete_deferrals()
            result = cast(
                "IntentGraphState",
                {
                    IntentStateKey.VERIFY_MODE: None,
                    CommonStateKey.IS_COMPLETE: True,
                    IntentStateKey.SHOULD_RETRY: False,
                    CommonStateKey.COMPLETION_REASON: CompletionReason.MAX_STEPS.value,
                },
            )
            self.__provider.persistence.persist(result=result)
            return result

        # Use type guard to satisfy MyPy
        screen_capture = state.get(CommonStateKey.CAPTURE)

        if not screen_capture or not isinstance(screen_capture, ScreenCapture):
            return self.__handle_no_capture()

        capture: ScreenCapture = screen_capture

        # ERROR BOUNDARY: Wrap planning logic
        try:
            # Check injected context
            current_step = self.__provider.context.agent_state.step_count
            state_injected = state.get(IntentStateKey.INJECTED_CONTEXT)
            guidance_snapshot = self.__provider.context.context_manager.get_user_guidance()

            logger.info(
                f"[H3] Analysis Context | Step: {current_step} | "
                f"Active Guidance: {len(guidance_snapshot)} items | "
                f"State Injected: {state_injected is not None}"
            )

            start_time = time.time()
            raw_elements = state.get(IntentStateKey.ELEMENTS)

            elements: Optional[Dict[str, Any]] = None
            if isinstance(raw_elements, dict):
                elements = raw_elements

            # Use orientation-corrected capture dims so the planner's prompt
            # advertises the actual canvas the screenshot was rendered on, ot the device-cached portrait template.
            width = capture.width
            height = capture.height

            # Domain capabilities flow through AgentState;
            # only the configuration toggle for the stuck-loop HITL synthesis needs explicit threading.
            prompt_if_stuck = self.__provider.context.configuration.intent.prompt_user_if_stuck

            # HITL: Check for pause request or context injection before planning
            await self.__provider.hitl.prompt(step=current_step)

            logger.info(
                f"Calling planner for step {current_step + 1}",
                extra={
                    "event": "analyze.log",
                    "component": "graph.intent.analyze",
                    "workflow.id": self.__provider.context.workflow_id,
                },
            )
            observation = state.get(CommonStateKey.SCREEN_OBSERVATION)

            plan = await self.__provider.context.planner.plan_step(
                capture=capture,
                elements=elements,
                screen_observation=(
                    observation if isinstance(observation, ScreenObservation) else None
                ),
                screen_width=width,
                screen_height=height,
                prompt_if_stuck=prompt_if_stuck,
                use_xml=self.__provider.context.use_xml,
                reasoner=self.__provider.context.reasoner,
                state=self.__provider.context.agent_state,
                context_manager=self.__provider.context.context_manager,
            )

            duration = time.time() - start_time
            self.__provider.context.metrics.record(operation="analysis", duration=duration)

            if plan.metrics:
                self.__provider.context.metrics.record_tokens(
                    prompt=int(plan.metrics.get("prompt_tokens", 0)),
                    cached=int(plan.metrics.get("cached_tokens", 0)),
                    completion=int(plan.metrics.get("completion_tokens", 0)),
                )

            # Log plan details
            if plan.step:
                logger.info(
                    f"Plan created: action={plan.step.action.action_type.value}, "
                    f"confidence={plan.step.action.confidence:.2f}, "
                    f"target={plan.step.action.target}"
                )

                # Emit structured telemetry for streaming UI
                await self.__provider.context.telemetry.info(
                    plan.reason or "No reasoning",
                    step=current_step + 1,
                    reasoning=plan.reason,
                    type=FathomEvent.REASONING,
                    rationale=plan.step.action.rationale if plan.step else None,
                )
                await self.__provider.context.telemetry.info(
                    plan.step.action.to_description(),
                    type=FathomEvent.PLANNED_ACTION,
                    step=current_step + 1,
                )
            else:
                logger.warning(
                    f"No step planned: is_complete={plan.is_complete}, "
                    f"should_retry={plan.should_retry}, reason={plan.reason}"
                )

            logger.info(
                f"Analysis completed in {duration:.2f}s: "
                f"is_complete={plan.is_complete}, should_retry={plan.should_retry}, "
                f"has_step={plan.step is not None}"
            )

            completion_reason = self.__completion_reason(
                plan_reason=plan.reason,
                plan_complete=plan.is_complete,
                graph_reason=state.get(CommonStateKey.COMPLETION_REASON),
            )
            # Bounded-retry deferral when sub-goals are still open. Owned by
            # ANALYZE (not the router) so the counter increment lands inside
            # the persisted checkpoint below — a router-side mutation would
            # be overwritten by the next ``persistence.restore()``.
            effective_is_complete = plan.is_complete
            effective_completion_reason = completion_reason
            agent_state = self.__provider.context.agent_state
            completion_deferred = False

            if (
                plan.is_complete
                and completion_reason == CompletionReason.SUCCESS.value
                and agent_state.has_sub_goals()
                and not agent_state.all_sub_goals_complete()
            ):
                deferrals = agent_state.record_complete_deferral()
                if deferrals <= DEFAULT_COMPLETE_DEFERRAL_BUDGET:
                    logger.info(
                        "Deferring planner completion; sub-goals remain",
                        extra={
                            "deferrals": deferrals,
                            "component": "graph.intent.analyze",
                            "event": "analyze.complete.deferred",
                            "budget": DEFAULT_COMPLETE_DEFERRAL_BUDGET,
                            "workflow.id": self.__provider.context.workflow_id,
                        },
                    )
                    agent_state.reset_completion()
                    effective_is_complete = False
                    completion_deferred = True
                    effective_completion_reason = None
                else:
                    logger.warning(
                        "Complete-deferral budget exhausted; honouring planner verdict",
                        extra={
                            "deferrals": deferrals,
                            "component": "graph.intent.analyze",
                            "budget": DEFAULT_COMPLETE_DEFERRAL_BUDGET,
                            "event": "analyze.complete.budget_exhausted",
                            "workflow.id": self.__provider.context.workflow_id,
                        },
                    )
                    agent_state.reset_complete_deferrals()
            else:
                # Either no sub-goals, all complete, or non-complete plan — clear any stale streak.
                agent_state.reset_complete_deferrals()

            result = cast(
                "IntentGraphState",
                {
                    IntentStateKey.PLAN: plan,
                    IntentStateKey.ELEMENTS: elements,
                    IntentStateKey.INJECTED_CONTEXT: None,
                    IntentStateKey.VERIFY_MODE: self.__verify_mode(
                        is_complete=effective_is_complete,
                        completion_reason=effective_completion_reason,
                    ),
                    IntentStateKey.PLANNED_STEP: None if completion_deferred else plan.step,
                    CommonStateKey.ANALYSIS_DURATION: duration,
                    IntentStateKey.SHOULD_RETRY: (
                        True
                        if completion_deferred
                        else (False if effective_is_complete else plan.should_retry)
                    ),
                    CommonStateKey.IS_COMPLETE: effective_is_complete,
                    CommonStateKey.COMPLETION_REASON: effective_completion_reason,
                    CommonStateKey.SCREEN_OBSERVATION: state.get(CommonStateKey.SCREEN_OBSERVATION),
                },
            )

            # Log what will happen next based on routing logic
            if effective_is_complete:
                destination = self.__completion_destination(
                    completion_reason=effective_completion_reason
                )
                logger.info(
                    "Analyze completion route selected",
                    extra={
                        "event": "analyze.log",
                        "route.destination": destination,
                        "completion.reason": effective_completion_reason,
                        "component": "graph.intent.analyze",
                        "workflow.id": self.__provider.context.workflow_id,
                    },
                )

            elif plan.is_complete:
                logger.info(
                    "-> Will route to GROUND (is_complete deferred; sub-goals remain)",
                    extra={
                        "event": "analyze.log",
                        "component": "graph.intent.analyze",
                        "workflow.id": self.__provider.context.workflow_id,
                    },
                )

            elif plan.should_retry:
                terminate_result = self.__consume_planner_retry(
                    base_result=result,
                    plan_metadata=plan.metadata,
                )
                if terminate_result is not None:
                    self.__provider.persistence.persist(result=terminate_result)
                    return terminate_result

                logger.info(
                    "-> Will route to GROUND (should_retry=True)",
                    extra={
                        "event": "analyze.log",
                        "component": "graph.intent.analyze",
                        "workflow.id": self.__provider.context.workflow_id,
                    },
                )

            elif not plan.step:
                logger.info(
                    "-> Will route to GROUND (no planned_step)",
                    extra={
                        "event": "analyze.log",
                        "component": "graph.intent.analyze",
                        "workflow.id": self.__provider.context.workflow_id,
                    },
                )

            else:
                logger.info(
                    "-> Will route to EXECUTE",
                    extra={
                        "event": "analyze.log",
                        "component": "graph.intent.analyze",
                        "workflow.id": self.__provider.context.workflow_id,
                    },
                )

            # Persist sub-goal state to graph for checkpoint recovery
            self.__provider.persistence.persist(result=result)

            return result

        except asyncio.CancelledError:
            # CancelledError is the cooperative cancellation signal; it
            # must propagate so the LangGraph task tree unwinds cleanly.
            raise
        except Exception as exception:
            logger.exception(
                f"Analysis failed: {exception}",
                extra={
                    "event": "analyze.log",
                    "component": "graph.intent.analyze",
                    "workflow.id": self.__provider.context.workflow_id,
                },
            )
            await self.__provider.context.telemetry.error(
                f"Analysis failed: {exception}",
                step=self.__provider.context.agent_state.step_count + 1,
            )
            self.__provider.context.agent_state.mark_complete(reason=CompletionReason.FAILED.value)
            diagnostic = f"Analysis failed: {exception}"[:500]
            result = cast(
                "IntentGraphState",
                {
                    IntentStateKey.VERIFY_MODE: None,
                    IntentStateKey.SHOULD_RETRY: False,
                    CommonStateKey.ANALYSIS_DURATION: 0.0,
                    IntentStateKey.INJECTED_CONTEXT: None,
                    CommonStateKey.IS_COMPLETE: True,
                    CommonStateKey.COMPLETION_REASON: CompletionReason.FAILED.value,
                    CommonStateKey.FAILURE_DIAGNOSTIC: diagnostic,
                },
            )
            self.__provider.persistence.persist(result=result)

            return result

    def __handle_no_capture(self) -> IntentGraphState:
        """
        Terminate the workflow when GROUND cannot produce a valid ScreenCapture.
        """

        agent_state = self.__provider.context.agent_state

        logger.error(
            "No valid screen capture from GROUND; terminating workflow",
            extra={
                "event": "analyze.no_capture",
                "component": "graph.intent.analyze",
                "workflow.id": self.__provider.context.workflow_id,
            },
        )
        agent_state.mark_complete(reason=CompletionReason.FAILED.value)
        agent_state.reset_complete_deferrals()
        terminate = cast(
            "IntentGraphState",
            {
                IntentStateKey.VERIFY_MODE: None,
                CommonStateKey.IS_COMPLETE: True,
                IntentStateKey.SHOULD_RETRY: False,
                IntentStateKey.INJECTED_CONTEXT: None,
                CommonStateKey.COMPLETION_REASON: CompletionReason.FAILED.value,
                CommonStateKey.FAILURE_DIAGNOSTIC: (
                    "GROUND produced no valid ScreenCapture; transient capture "
                    "retries belong inside the device adapter, not here."
                ),
            },
        )
        self.__provider.persistence.persist(result=terminate)
        return terminate

    def __verify_mode(
        self,
        *,
        is_complete: bool,
        completion_reason: Optional[str],
    ) -> Optional[str]:
        """
        Return the VERIFY mode only for non-terminal completion routing.
        """

        if not is_complete or completion_reason in TERMINAL_COMPLETION_REASONS:
            return None

        return self.__verification_modes.mode_for_producer(
            agent_state=self.__provider.context.agent_state
        ).value

    @staticmethod
    def __completion_reason(
        *, plan_reason: str, plan_complete: bool, graph_reason: object
    ) -> Optional[str]:
        """
        Return the typed completion reason for this analyze result.
        """

        if plan_complete:
            return plan_reason

        return graph_reason if isinstance(graph_reason, str) else None

    @staticmethod
    def __completion_destination(*, completion_reason: Optional[str]) -> str:
        """
        Return the router destination label for a completion reason.
        """

        if completion_reason in TERMINAL_COMPLETION_REASONS:
            return "END"

        return "VERIFY"

    def __consume_planner_retry(
        self,
        *,
        base_result: IntentGraphState,
        plan_metadata: Optional[Dict[str, Any]],
    ) -> Optional[IntentGraphState]:
        """
        Consume planner-retry budget; return a terminal state on exhaustion or None to continue.
        """

        metadata = plan_metadata or {}
        agent_state = self.__provider.context.agent_state

        kind = self.__coerce_retry_kind(raw=metadata.get(RetryMetadataField.KIND.value))
        branch = self.__coerce_retry_branch(raw=metadata.get(RetryMetadataField.BRANCH.value))

        # ESCALATION_DEFERRED is bounded by the per-sub-goal ``deferral_count``
        # when a sub-goal is active; bypass the planner-retry budget in that
        # case. With no active sub-goal there's no deferral_count to bound it,
        # so fall through to ``tick_planner_retry`` (which knows to count it).
        if kind is RetryKind.ESCALATION_DEFERRED and agent_state.get_current_sub_goal() is not None:
            return None

        action = metadata.get(RetryMetadataField.BLOCKED_ACTION.value)
        action = action if isinstance(action, str) else None

        count = agent_state.tick_planner_retry(kind=kind, branch=branch, action=action)
        planner = agent_state.retries.planner

        logger.info(
            "Planner-retry budget consumed",
            extra={
                "event": PLANNER_RETRY_CONSUMED,
                "component": "graph.intent.analyze",
                "retry.kind": kind.value,
                "action.descriptor": action,
                "retry.branch": branch.value,
                "retries.planner.count": count,
                "retries.planner.cap": planner.cap,
                "step.count": agent_state.step_count,
                "workflow.id": self.__provider.context.workflow_id,
            },
        )

        if not planner.exhausted:
            return None

        logger.error(
            "Planner-retry budget exhausted; terminating workflow",
            extra={
                "event": PLANNER_RETRY_EXHAUSTED,
                "component": "graph.intent.analyze",
                "retry.kind": kind.value,
                "retry.branch": branch.value,
                "retries.planner.count": count,
                "retries.planner.cap": planner.cap,
                "step.count": agent_state.step_count,
                "workflow.id": self.__provider.context.workflow_id,
            },
        )
        agent_state.mark_complete(reason=CompletionReason.RETRY_BUDGET_EXHAUSTED.value)
        terminate = dict(base_result)

        terminate[CommonStateKey.IS_COMPLETE] = True
        terminate[IntentStateKey.VERIFY_MODE] = None
        terminate[IntentStateKey.SHOULD_RETRY] = False
        terminate[CommonStateKey.COMPLETION_REASON] = CompletionReason.RETRY_BUDGET_EXHAUSTED.value

        return cast("IntentGraphState", terminate)

    def __coerce_retry_kind(self, *, raw: object) -> RetryKind:
        """
        Map planner-stamped KIND metadata onto :class:`RetryKind`; warn and default to ``SILENT_REJECTION`` on missing or unrecognized values.
        """

        if isinstance(raw, str):
            try:
                return RetryKind(raw)
            except ValueError:
                self.__log_invalid_metadata(
                    raw=raw,
                    field=RetryMetadataField.KIND.value,
                    default=RetryKind.SILENT_REJECTION.value,
                )
                return RetryKind.SILENT_REJECTION

        self.__log_invalid_metadata(
            raw=raw,
            field=RetryMetadataField.KIND.value,
            default=RetryKind.SILENT_REJECTION.value,
        )
        return RetryKind.SILENT_REJECTION

    def __coerce_retry_branch(self, *, raw: object) -> RetryBranch:
        """
        Map planner-stamped BRANCH metadata onto :class:`RetryBranch`; warn and default to ``UNKNOWN`` on missing or unrecognized values.
        """

        if isinstance(raw, str):
            try:
                return RetryBranch(raw)
            except ValueError:
                self.__log_invalid_metadata(
                    raw=raw,
                    default=RetryBranch.UNKNOWN.value,
                    field=RetryMetadataField.BRANCH.value,
                )
                return RetryBranch.UNKNOWN

        self.__log_invalid_metadata(
            raw=raw,
            default=RetryBranch.UNKNOWN.value,
            field=RetryMetadataField.BRANCH.value,
        )
        return RetryBranch.UNKNOWN

    def __log_invalid_metadata(self, *, field: str, raw: object, default: str) -> None:
        """
        Emit a structured warning whenever planner-retry metadata is missing or unrecognized so contract drift surfaces in logs.
        """

        logger.warning(
            "Invalid planner-retry metadata; falling back to default",
            extra={
                "event": PLANNER_RETRY_METADATA_INVALID,
                "component": "graph.intent.analyze",
                "metadata.field": field,
                "metadata.raw": repr(raw),
                "metadata.default": default,
                "workflow.id": self.__provider.context.workflow_id,
            },
        )
