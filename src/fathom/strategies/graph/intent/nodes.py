from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from typing import Any, Callable, Dict, Optional

from fathom.constants import ActionType
from fathom.constants.execution import VISUAL_HASH_LENGTH
from fathom.constants.graph import NodeName
from fathom.schemas.results import ActionResult, AnalysisResult, PlanResult
from fathom.schemas.screens import ScreenCapture, ScreenState
from fathom.schemas.steps import Step, StepResult
from fathom.strategies.graph.context import GraphContext
from fathom.strategies.graph.state import IntentGraphState

logger = logging.getLogger(__name__)


class IntentNodeProvider:
    """
    Provides LangGraph nodes for intent execution.
    Encapsulates dependencies and shared private logic.
    """

    def __init__(self, context: GraphContext) -> None:
        """
        Initialize provider with shared context.
        """

        self.__context = context

    async def ground(self, state: IntentGraphState) -> IntentGraphState:
        """
        Capture the screen and update state.
        """
        if self.__context.is_cancelled:
            return {
                "is_complete": True,
                "completion_reason": "Cancelled",
            }

        start_time = time.time()

        try:
            # Parallel Snapshot (Screenshot + XML) via DevicePort
            screenshot_bytes, xml_content = await self.__context.device.get_snapshot()

            if not screenshot_bytes or len(screenshot_bytes) == 0:
                self.__context.telemetry.error("Ground: Empty screenshot captured")
                return {
                    "capture": None,
                    "completion_reason": "Empty screenshot captured",
                    "is_complete": True,
                }

            width, height = await self.__context.device.get_dimensions()

            # Validate dimensions
            if width <= 0 or height <= 0:
                self.__context.telemetry.error(f"Ground: Invalid dimensions {width}x{height}")
                return {
                    "capture": None,
                    "completion_reason": f"Invalid dimensions: {width}x{height}",
                    "is_complete": True,
                }

            # Get current package
            try:
                activity = await self.__context.device.get_current_package()
            except Exception as exception:
                self.__context.telemetry.warning(
                    f"Ground: Failed to get current package: {exception}"
                )
                activity = "unknown"

            # Persist capture
            storage_id = await self.__context.storage.save(
                data=screenshot_bytes,
                metadata={
                    "type": "screenshots",
                    "activity_name": activity,
                    "package_name": activity,
                    "session_id": self.__context.workflow_id,
                    "timestamp": time.time(),
                },
            )

            screen = ScreenCapture(
                image=screenshot_bytes,
                width=width,
                height=height,
                activity=activity,
                timestamp=int(time.time() * 1000),
                metadata={"storage_id": storage_id},
            )

            # XML Dump if enabled
            xml_content_str = xml_content if xml_content else None
            elements = None

            if self.__context.use_xml and xml_content_str:
                dump_start = time.time()
                self.__context.metrics.record(
                    operation="hierarchy_dump", duration=time.time() - dump_start
                )

                process_start = time.time()
                (
                    annotated_screen,
                    elements,
                ) = await self.__context.hierarchy.process_xml_and_screen(
                    screen=screen,
                    xml=xml_content_str,
                    path_manager=self.__context.path_manager,
                    package_name=activity,
                    session_id=self.__context.workflow_id,
                    action_type=ActionType.TAP,
                )
                self.__context.metrics.record(
                    operation="hierarchy_processing", duration=time.time() - process_start
                )

                if annotated_screen:
                    screen = annotated_screen

            # Update Agent State
            visual_hash = hashlib.sha256(screen.image).hexdigest()[:VISUAL_HASH_LENGTH]

            screen_state = ScreenState(
                visual_hash=visual_hash,
                activity=screen.activity,
                timestamp=screen.timestamp,
                activity_hash=hashlib.md5(
                    screen.activity.encode(), usedforsecurity=False
                ).hexdigest()[:VISUAL_HASH_LENGTH],
                structural_hash="0" * VISUAL_HASH_LENGTH,
            )

            is_new_screen = self.__context.agent_state.update_screen(screen=screen_state)

            duration = time.time() - start_time
            self.__context.metrics.record(operation="screenshot", duration=duration)

            # Reset per-step fields
            return {
                "capture": screen,
                "screen_state": screen_state,
                "is_new_screen": is_new_screen,
                "xml_content": xml_content_str,
                "elements": elements,
                "grounding_duration": duration,
                "planned_step": None,
                "step_result": None,
                "analysis": None,
            }

        except Exception as exception:
            self.__context.telemetry.error(f"Grounding failed: {exception}")
            return {
                "capture": None,
                "completion_reason": f"Grounding failed: {exception}",
                "is_complete": True,
            }

    async def analyze(self, state: IntentGraphState) -> IntentGraphState:
        """
        Plan the next step using the Agent Planner.
        """
        if self.__context.is_cancelled:
            return {"is_complete": True}

        # Use type guard to satisfy MyPy
        screen_capture = state.get("capture")
        if not screen_capture or not isinstance(screen_capture, ScreenCapture):
            return {"should_retry": True}

        capture: ScreenCapture = screen_capture

        # Check injected context
        state_injected = state.get("injected_context")
        current_step = self.__context.agent_state.step_count

        guidance_snapshot = self.__context.context_manager.get_user_guidance()
        logger.debug(
            f"[H3] Analysis Context | Step: {current_step} | "
            f"Active Guidance: {len(guidance_snapshot)} items | "
            f"State Injected: {state_injected is not None}"
        )

        start_time = time.time()

        raw_elements = state.get("elements")
        elements: Optional[Dict[str, Any]] = None
        if isinstance(raw_elements, dict):
            elements = raw_elements

        plan = await self.__context.planner.plan_step(
            state=self.__context.agent_state,
            reasoner=self.__context.reasoner,
            use_xml=self.__context.use_xml,
            capture=capture,
            elements=elements,
            context_manager=self.__context.context_manager,
        )

        duration = time.time() - start_time
        self.__context.metrics.record(operation="analysis", duration=duration)

        if plan.metrics:
            self.__context.metrics.record_tokens(
                prompt=int(plan.metrics.get("prompt_tokens", 0)),
                completion=int(plan.metrics.get("completion_tokens", 0)),
                cached=int(plan.metrics.get("cached_tokens", 0)),
            )

        if plan.step:
            self.__context.ux.render_fallback(
                reasoning=plan.reason or "No reasoning",
                action=plan.step.action.to_description(),
                step_number=self.__context.agent_state.step_count + 1,
            )

        return {
            "plan": plan,
            "planned_step": plan.step,
            "is_complete": plan.is_complete,
            "completion_reason": plan.reason
            if plan.is_complete
            else state.get("completion_reason"),
            "should_retry": plan.should_retry,
            "analysis_duration": duration,
            "injected_context": None,
        }

    async def execute(self, state: IntentGraphState) -> IntentGraphState:
        """
        Execute the planned action via ActionExecutor.
        """
        if self.__context.is_cancelled:
            return {"is_complete": True}

        # Type guards for planned_step and capture
        current_step = state.get("planned_step")
        screen_capture = state.get("capture")

        if not isinstance(current_step, Step) or not isinstance(screen_capture, ScreenCapture):
            return {}

        step: Step = current_step
        capture: ScreenCapture = screen_capture

        start_time = time.time()

        # Resolve References
        resolved_action = await self.__context.resolution.resolve(action=step.action)
        step = step.model_copy(update={"action": resolved_action})

        # Determine package name from state for tracing
        package_name = "unknown"
        current_screen = state.get("screen_state")
        if isinstance(current_screen, ScreenState) and current_screen.activity:
            package_name = current_screen.activity

        # Delegate to ActionExecutor
        execution_result = await self.__context.action_executor.act(
            step=step,
            pre_capture=capture,
            package_name=package_name,
            session_id=self.__context.workflow_id,
        )

        # Wait for screen stability after action
        await asyncio.sleep(delay=self.__context.configuration.engine.stability_wait)

        # Capture post-execution screen to compute post_hash
        try:
            post_screenshot, _ = await self.__context.device.get_snapshot()
            post_hash = (
                hashlib.sha256(post_screenshot).hexdigest()[:VISUAL_HASH_LENGTH]
                if post_screenshot
                else "0"
            )
        except Exception as exception:
            self.__context.telemetry.warning(f"Execute: Failed to capture post-screen: {exception}")
            post_hash = "0"

        duration = time.time() - start_time
        self.__context.metrics.record(operation="action", duration=duration)

        current_screen_state = state.get("screen_state")
        pre_hash = (
            current_screen_state.visual_hash
            if isinstance(current_screen_state, ScreenState)
            else "0"
        )

        screen_changed = pre_hash != post_hash

        step_result = StepResult(
            step=step,
            pre_hash=pre_hash,
            post_hash=post_hash,
            success=execution_result.success,
            duration=int(duration * 1000),
            error=execution_result.error,
            screen_changed=screen_changed,
        )

        return {
            "step_result": step_result,
            "execution_duration": duration,
        }

    async def record(self, state: IntentGraphState) -> IntentGraphState:
        """
        Record the execution result.
        """
        if self.__context.is_cancelled:
            return {"is_complete": True}

        result = state.get("step_result")
        if not isinstance(result, StepResult):
            return {}

        step_result: StepResult = result

        self.__context.agent_state.record_step(result=step_result)
        self.__context.history.save_step(result=step_result, intent=self.__context.intent)

        await self.__context.memory.store_experience(
            visual_hash=step_result.pre_hash,
            action=step_result.step.action,
            success=step_result.success,
        )

        # Commit cycle to ContextManager (GCC Trace)
        logger.debug(
            f"[H3] Committing to trace | thought={step_result.step.action.rationale[:50]}..."
        )

        analysis_result = state.get("analysis")
        analysis: Optional[AnalysisResult] = None
        if isinstance(analysis_result, AnalysisResult):
            analysis = analysis_result

        observation = f"Screen: {step_result.pre_hash[:8]}"
        if analysis and analysis.screen_description:
            observation += f" | Content: {analysis.screen_description[:100]}..."

        await self.__context.context_manager.commit(
            observation=observation,
            thought=step_result.step.action.rationale,
            action=step_result.step.action,
        )

        # GCC Branching
        full_context = self.__context.context_manager.get_full_context()
        trace = full_context.get("trace", [])
        if len(trace) >= 5:
            await self.__context.context_manager.branch()

        # Audit logging
        execution_plan = state.get("plan")
        current_screen = state.get("screen_state")
        is_new_screen = state.get("is_new_screen")

        if (
            isinstance(execution_plan, PlanResult)
            and isinstance(current_screen, ScreenState)
            and isinstance(is_new_screen, bool)
        ):
            plan: PlanResult = execution_plan
            screen_state: ScreenState = current_screen
            is_new: bool = is_new_screen

            # Explicit float conversion for metrics
            analysis_duration = float(state.get("analysis_duration") or 0.0)
            grounding_duration = float(state.get("grounding_duration") or 0.0)
            execution_duration = float(state.get("execution_duration") or 0.0)

            self.__context.auditor.log_step(
                plan=plan,
                state=screen_state,
                result=ActionResult(success=step_result.success, duration=step_result.duration),
                is_new_screen=is_new,
                is_stuck=self.__context.agent_state.is_stuck,
                step_count=self.__context.agent_state.step_count,
                analysis_duration=analysis_duration,
                grounding_duration=grounding_duration,
                hierarchy_duration=0.0,
                execution_duration=execution_duration,
                total_duration=grounding_duration + analysis_duration + execution_duration,
            )

        # Check max steps
        if self.__context.agent_state.step_count >= self.__context.max_steps:
            self.__context.agent_state.mark_complete(reason="Max steps reached")
            return {
                "is_complete": True,
                "completion_reason": "Max steps reached",
            }

        return {}


class IntentGraphFactory:
    """
    Factory for building the Intent Node functions.
    Wraps node logic to prevent standalone functions and cleaner imports.
    """

    @staticmethod
    def build(context: GraphContext) -> Dict[str, Callable[..., Any]]:
        """
        Builds the node functions for the intent graph.
        """
        provider = IntentNodeProvider(context=context)

        return {
            NodeName.GROUND: provider.ground,
            NodeName.ANALYZE: provider.analyze,
            NodeName.EXECUTE: provider.execute,
            NodeName.RECORD: provider.record,
        }
