from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, Optional, cast

from fathom.constants import FathomEvent
from fathom.constants.runtime import DEFAULT_COMPLETE_DEFERRAL_BUDGET
from fathom.constants.state import CommonStateKey, CompletionReason, IntentStateKey
from fathom.schemas.observation import ScreenObservation
from fathom.schemas.screens import ScreenCapture
from fathom.strategies.graph.intent.nodes.provider import IntentNodeProvider
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

            result = cast(
                "IntentGraphState",
                {
                    CommonStateKey.IS_COMPLETE: True,
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
            result = cast(
                "IntentGraphState",
                {
                    CommonStateKey.IS_COMPLETE: True,
                    CommonStateKey.COMPLETION_REASON: CompletionReason.MAX_STEPS.value,
                },
            )
            self.__provider.persistence.persist(result=result)
            return result

        # Use type guard to satisfy MyPy
        screen_capture = state.get(CommonStateKey.CAPTURE)

        if not screen_capture or not isinstance(screen_capture, ScreenCapture):
            logger.error(
                "No valid screen capture found, setting should_retry=True",
                extra={
                    "event": "analyze.log",
                    "component": "graph.intent.analyze",
                    "workflow.id": self.__provider.context.workflow_id,
                },
            )
            return cast("IntentGraphState", {IntentStateKey.SHOULD_RETRY: True})

        capture: ScreenCapture = screen_capture

        # ERROR BOUNDARY: Wrap planning logic
        try:
            # Check injected context
            current_step = self.__provider.context.agent_state.step_count
            state_injected = state.get(IntentStateKey.INJECTED_CONTEXT)
            guidance_snapshot = self.__provider.context.context_manager.get_user_guidance()

            logger.debug(
                f"[H3] Analysis Context | Step: {current_step} | "
                f"Active Guidance: {len(guidance_snapshot)} items | "
                f"State Injected: {state_injected is not None}"
            )

            start_time = time.time()
            raw_elements = state.get(IntentStateKey.ELEMENTS)

            elements: Optional[Dict[str, Any]] = None
            if isinstance(raw_elements, dict):
                elements = raw_elements

            # Get Device Dimensions for Accurate Normalization (Strict)
            width, height = await self.__provider.context.device.get_dimensions()

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

            completion_reason = (
                plan.reason if plan.is_complete else state.get(CommonStateKey.COMPLETION_REASON)
            )
            # Bounded-retry deferral when sub-goals are still open. Owned by
            # ANALYZE (not the router) so the counter increment lands inside
            # the persisted checkpoint below — a router-side mutation would
            # be overwritten by the next ``persistence.restore()``.
            effective_is_complete = plan.is_complete
            effective_completion_reason = completion_reason
            agent_state = self.__provider.context.agent_state

            if (
                plan.is_complete
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
                    IntentStateKey.PLANNED_STEP: plan.step,
                    CommonStateKey.ANALYSIS_DURATION: duration,
                    IntentStateKey.SHOULD_RETRY: plan.should_retry,
                    CommonStateKey.IS_COMPLETE: effective_is_complete,
                    CommonStateKey.COMPLETION_REASON: effective_completion_reason,
                    CommonStateKey.SCREEN_OBSERVATION: state.get(CommonStateKey.SCREEN_OBSERVATION),
                },
            )

            # Log what will happen next based on routing logic
            if effective_is_complete:
                logger.info(
                    "-> Will route to VERIFY (is_complete=True)",
                    extra={
                        "event": "analyze.log",
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
            result = cast(
                "IntentGraphState",
                {
                    IntentStateKey.SHOULD_RETRY: False,
                    CommonStateKey.ANALYSIS_DURATION: 0.0,
                    IntentStateKey.INJECTED_CONTEXT: None,
                    CommonStateKey.IS_COMPLETE: True,
                    CommonStateKey.COMPLETION_REASON: CompletionReason.FAILED.value,
                    CommonStateKey.FAILURE_DIAGNOSTIC: f"Analysis failed: {exception}",
                },
            )
            self.__provider.persistence.persist(result=result)

            return result
