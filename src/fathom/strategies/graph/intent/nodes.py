from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from typing import Any, Callable, Dict, Optional, Tuple

from fathom.constants import ActionType
from fathom.constants.execution import VISUAL_HASH_LENGTH
from fathom.constants.graph import NodeName
from fathom.schemas.actions import Bounds
from fathom.schemas.results import ActionResult, AnalysisResult, PlanResult
from fathom.schemas.screens import ScreenCapture, ScreenState
from fathom.schemas.steps import Step, StepResult
from fathom.strategies.graph.context import GraphContext
from fathom.strategies.graph.state import IntentGraphState
from fathom.utils.coordinates import CoordinateConverter

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

        _ = state

        if self.__context.is_cancelled:
            return {"is_complete": True, "completion_reason": "Cancelled"}

        start_time = time.time()

        try:
            # Parallel Snapshot (Screenshot + XML) via DevicePort
            screenshot_bytes, xml_content = await self.__context.device.get_snapshot()

            if not screenshot_bytes:
                return {
                    "capture": None,
                    "is_complete": False,
                    "completion_reason": "Empty screenshot captured",
                }

            width, height = await self.__context.device.get_dimensions()

            # Get current package
            try:
                activity = await self.__context.device.get_current_package()
            except Exception as exception:
                activity = "unknown"
                self.__context.telemetry.warning(
                    "Failed to get current package", error=str(exception)
                )

            # Persist capture
            storage_id = await self.__context.storage.save(
                data=screenshot_bytes,
                metadata={
                    "type": "screenshots",
                    "package_name": activity,
                    "timestamp": time.time(),
                    "activity_name": activity,
                    "session_id": self.__context.workflow_id,
                },
            )

            screen = ScreenCapture(
                width=width,
                height=height,
                activity=activity,
                image=screenshot_bytes,
                timestamp=int(time.time() * 1000),
                metadata={"storage_id": storage_id},
            )

            # XML Dump if enabled
            elements = None
            xml_content_str = xml_content if xml_content else None

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
                    package_name=activity,
                    action_type=ActionType.TAP,
                    session_id=self.__context.workflow_id,
                    path_manager=self.__context.path_manager,
                )
                self.__context.metrics.record(
                    operation="hierarchy_processing", duration=time.time() - process_start
                )

                if annotated_screen:
                    screen = annotated_screen

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
                "analysis": None,
                "capture": screen,
                "step_result": None,
                "elements": elements,
                "planned_step": None,
                "screen_state": screen_state,
                "is_new_screen": is_new_screen,
                "xml_content": xml_content_str,
                "grounding_duration": duration,
            }

        except Exception as exception:
            self.__context.telemetry.error(f"Grounding failed: {exception}")
            return {
                "capture": None,
                "is_complete": False,
                "completion_reason": f"Grounding failed: {exception}",
            }

    async def analyze(self, state: IntentGraphState) -> IntentGraphState:
        """
        Plan the next step using the Agent Planner.
        """

        if self.__context.is_cancelled:
            return {"is_complete": True}

        capture: Optional[ScreenCapture] = state.get("capture")

        if not capture:
            return {"should_retry": True}

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

        plan = await self.__context.planner.plan_step(
            capture=capture,
            use_xml=self.__context.use_xml,
            elements=state.get("elements"),
            state=self.__context.agent_state,
            reasoner=self.__context.reasoner,
            context_manager=self.__context.context_manager,
        )

        duration = time.time() - start_time
        self.__context.metrics.record(operation="analysis", duration=duration)

        if plan.metrics:
            self.__context.metrics.record_tokens(
                cached=int(plan.metrics.get("cached_tokens", 0)),
                prompt=int(plan.metrics.get("prompt_tokens", 0)),
                completion=int(plan.metrics.get("completion_tokens", 0)),
            )

        if plan.step:
            self.__context.ux.render_fallback(
                reasoning=plan.reason or "No reasoning",
                action=plan.step.action.to_description(),
                step_number=self.__context.agent_state.step_count + 1,
            )

        completion_reason = plan.reason if plan.is_complete else state.get("completion_reason")

        return {
            "plan": plan,
            "injected_context": None,
            "planned_step": plan.step,
            "analysis_duration": duration,
            "is_complete": plan.is_complete,
            "should_retry": plan.should_retry,
            "completion_reason": completion_reason,
        }

    async def execute(self, state: IntentGraphState) -> IntentGraphState:
        """
        Execute the planned action.
        """

        if self.__context.is_cancelled:
            return {"is_complete": True}

        step: Optional[Step] = state.get("planned_step")
        capture: Optional[ScreenCapture] = state.get("capture")

        if not step or not capture:
            return {}

        start_time = time.time()

        # Resolve References
        resolved_action = await self.__context.resolution.resolve(action=step.action)
        step = step.model_copy(update={"action": resolved_action})

        # Physical Execution
        size = await self.__context.device.get_dimensions()
        converter = CoordinateConverter(screen_width=size[0], screen_height=size[1])

        action = step.action
        device_result = ActionResult(success=False, duration=0)
        coords: Optional[Tuple[int, ...]] = None

        try:
            if action.action_type == ActionType.TAP:
                if action.bounds:
                    coords = converter.center_to_pixels(bounds=action.bounds)
                    x, y = coords
                else:
                    x, y = size[0] // 2, size[1] // 2
                    coords = (x, y)
                device_result = await self.__context.device.tap(x=x, y=y)

            elif action.action_type == ActionType.TYPE:
                if action.bounds:
                    coords = converter.center_to_pixels(bounds=action.bounds)
                    x, y = coords
                    await self.__context.device.tap(x=x, y=y)
                device_result = await self.__context.device.type(text=action.text or "")

            elif "swipe" in action.action_type.value:
                direction = "up"
                if "_" in action.action_type.value:
                    direction = action.action_type.value.split("_")[1]

                bounds = action.bounds or Bounds(x=200, y=200, width=600, height=600)
                coords = converter.swipe_coordinates(bounds=bounds, direction=direction)
                x1, y1, x2, y2 = coords
                device_result = await self.__context.device.swipe(x1=x1, y1=y1, x2=x2, y2=y2)

            elif action.action_type == ActionType.SCROLL:
                x1, y1, x2, y2 = size[0] // 2, size[1] // 2 + 300, size[0] // 2, size[1] // 2 - 300
                coords = (x1, y1, x2, y2)
                device_result = await self.__context.device.swipe(x1=x1, y1=y1, x2=x2, y2=y2)

            elif action.action_type in (ActionType.BACK, ActionType.HOME):
                if action.action_type == ActionType.BACK:
                    device_result = await self.__context.device.back()
                else:
                    device_result = await self.__context.device.home()

            elif action.action_type == ActionType.WAIT:
                await asyncio.sleep((action.wait_duration or 1000) / 1000)
                device_result = ActionResult(success=True, duration=action.wait_duration or 1000)

            # Tracing
            if coords:
                screen_state: Optional[ScreenState] = state.get("screen_state")

                if screen_state and screen_state.activity:
                    package_name = screen_state.activity
                else:
                    package_name = "unknown"

                # Background persistence
                asyncio.create_task(
                    asyncio.to_thread(
                        self.__context.trace.save,
                        action=action,
                        coords=coords,
                        image_data=capture.image,
                        package_name=package_name,
                        session_id=self.__context.workflow_id,
                        step_number=self.__context.agent_state.step_count,
                    )
                )

        except Exception as exception:
            device_result = ActionResult(success=False, duration=0, error=str(exception))

        duration = time.time() - start_time
        self.__context.metrics.record(operation="action", duration=duration)

        current_screen_state: Optional[ScreenState] = state.get("screen_state")
        pre_hash = current_screen_state.visual_hash if current_screen_state else "0"

        step_result = StepResult(
            step=step,
            post_hash="0",
            pre_hash=pre_hash,
            screen_changed=True,
            error=device_result.error,
            success=device_result.success,
            duration=int(duration * 1000),
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

        step_result: Optional[StepResult] = state.get("step_result")
        if not step_result:
            return {}

        self.__context.agent_state.record_step(result=step_result)
        self.__context.history.save_step(result=step_result, intent=self.__context.intent)

        await self.__context.memory.store_experience(
            success=step_result.success,
            action=step_result.step.action,
            visual_hash=step_result.pre_hash,
        )

        # Commit cycle to ContextManager (GCC Trace)
        logger.debug(
            f"[H3] Committing to trace | thought={step_result.step.action.rationale[:50]}..."
        )

        analysis: Optional[AnalysisResult] = state.get("analysis")
        observation = f"Screen: {step_result.pre_hash[:8]}"
        if analysis and analysis.screen_description:
            observation += f" | Content: {analysis.screen_description[:100]}..."

        await self.__context.context_manager.commit(
            observation=observation,
            action=step_result.step.action,
            thought=step_result.step.action.rationale,
        )

        # GCC Branching
        full_context = self.__context.context_manager.get_full_context()
        trace = full_context.get("trace", [])

        if len(trace) >= 5:
            await self.__context.context_manager.branch()

        # Audit logging
        plan: Optional[PlanResult] = state.get("plan")
        is_new_screen: Optional[bool] = state.get("is_new_screen")
        screen_state: Optional[ScreenState] = state.get("screen_state")

        if plan and screen_state and is_new_screen is not None:
            analysis_duration: float = state.get("analysis_duration", 0.0)
            grounding_duration: float = state.get("grounding_duration", 0.0)
            execution_duration: float = state.get("execution_duration", 0.0)

            self.__context.auditor.log_step(
                plan=plan,
                state=screen_state,
                hierarchy_duration=0.0,
                is_new_screen=is_new_screen,
                analysis_duration=analysis_duration,
                grounding_duration=grounding_duration,
                execution_duration=execution_duration,
                is_stuck=self.__context.agent_state.is_stuck,
                step_count=self.__context.agent_state.step_count,
                total_duration=grounding_duration + analysis_duration + execution_duration,
                result=ActionResult(success=step_result.success, duration=step_result.duration),
            )

        # Check max steps
        if self.__context.agent_state.step_count >= self.__context.max_steps:
            self.__context.agent_state.mark_complete(reason="Max steps reached")
            return {
                "is_complete": True,
                "completion_reason": "Max steps reached",
            }

        return {}


def build_intent_nodes(context: GraphContext) -> Dict[str, Callable[..., Any]]:
    """
    Builds the node functions for the intent graph.
    """

    provider = IntentNodeProvider(context=context)

    return {
        NodeName.RECORD: provider.record,
        NodeName.GROUND: provider.ground,
        NodeName.ANALYZE: provider.analyze,
        NodeName.EXECUTE: provider.execute,
    }
