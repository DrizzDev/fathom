"""
Graph nodes for intent execution.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime
from typing import Any, Callable, Dict, Optional, Tuple

from fathom.constants import ActionType
from fathom.processing.annotator import ImageAnnotator
from fathom.schemas.actions import Bounds
from fathom.schemas.results import ActionResult
from fathom.schemas.screens import ScreenCapture, ScreenState
from fathom.schemas.steps import Step, StepResult
from fathom.strategies.graph.context import GraphContext
from fathom.strategies.graph.state import IntentGraphState
from fathom.utils.coordinates import CoordinateConverter


def build_intent_nodes(context: GraphContext) -> Dict[str, Callable[..., Any]]:
    """
    Builds the node functions for the intent graph.
    """

    async def _check_signal() -> Optional[str]:
        """Check for HITL signals and handle pause/resume."""
        signal = await context.signal.check_signal()
        if signal:
            await context.signal.wait_for_resume()
            if hasattr(context.signal, "get_injected_context"):
                return context.signal.get_injected_context()
        return None

    async def ground_node(state: IntentGraphState) -> IntentGraphState:
        """
        Capture the screen and update state.
        """
        if context.is_cancelled:
            return {**state, "is_complete": True, "completion_reason": "Cancelled"}

        # Check HITL signal
        injected = await _check_signal()
        if injected:
            await context.context_manager.inject_user_guidance(injected)
        
        start_time = time.time()

        try:
            # Capture screen via DevicePort
            screenshot_bytes = await context.device.capture_screen()
            
            if not screenshot_bytes:
                 return {
                    **state,
                    "capture": None,
                    "completion_reason": "Empty screenshot captured",
                    "is_complete": False, 
                }

            width, height = await context.device.get_screen_size()
            
            # Get current package
            try:
                activity = await context.device.get_current_package()
            except Exception:
                activity = "unknown"

            # Persist capture
            # Use dynamic activity (package) for folder structure
            storage_id = await context.storage.save(
                data=screenshot_bytes,
                metadata={
                    "type": "screenshots",
                    "activity_name": activity,
                    "package_name": activity,
                    "session_id": context.workflow_id,
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
            xml_content = None
            elements = None
            
            if context.use_xml:
                h_start = time.time()
                xml_content = await context.device.dump_hierarchy()
                h_dump_duration = time.time() - h_start
                context.metrics.record(operation="hierarchy_dump", duration=h_dump_duration)
                
                if xml_content:
                    # Process hierarchy
                    p_start = time.time()
                    annotated_screen, elements = await context.hierarchy.process_xml_and_screen(
                        screen=screen,
                        xml=xml_content,
                        path_manager=context.path_manager,
                        package_name=activity, # Use dynamic package
                        session_id=context.workflow_id,
                        action_type=ActionType.TAP,
                    )
                    p_duration = time.time() - p_start
                    context.metrics.record(operation="hierarchy_processing", duration=p_duration)
                    
                    if annotated_screen:
                        screen = annotated_screen

            # Update Agent State
            import hashlib
            from fathom.constants.execution import VISUAL_HASH_LENGTH
            
            visual_hash = hashlib.sha256(screen.image).hexdigest()[:VISUAL_HASH_LENGTH]
            
            from fathom.schemas.screens import ScreenState
            
            screen_state = ScreenState(
                visual_hash=visual_hash,
                activity=screen.activity,
                timestamp=screen.timestamp,
                activity_hash=hashlib.md5(
                    screen.activity.encode(), usedforsecurity=False
                ).hexdigest()[:VISUAL_HASH_LENGTH],
                structural_hash="0" * VISUAL_HASH_LENGTH,
            )
            
            is_new_screen = context.agent_state.update_screen(screen=screen_state)
            
            duration = time.time() - start_time
            context.metrics.record(operation="screenshot", duration=duration)
            
            return {
                **state,
                "capture": screen,
                "screen_state": screen_state,
                "is_new_screen": is_new_screen,
                "xml_content": xml_content,
                "elements": elements,
                "grounding_duration": duration,
                "injected_context": injected, 
                # Reset per-step
                "plan": None,
                "planned_step": None,
                "step_result": None,
            }

        except Exception as exception:
            context.telemetry.error(f"Grounding failed: {exception}")
            return {
                **state,
                "capture": None,
                "completion_reason": f"Grounding failed: {exception}",
                "is_complete": False, 
            }

    async def analyze_node(state: IntentGraphState) -> IntentGraphState:
        """
        Plan the next step using the Agent Planner.
        """
        if context.is_cancelled:
            return {**state, "is_complete": True}

        # Check HITL signal (local check)
        new_injected = await _check_signal()
        current_step = context.agent_state.step_count
        
        if new_injected:
            await context.context_manager.inject_user_guidance(guidance=new_injected, step=current_step)
            
        # Check injected context passed via state update
        state_injected = state.get("injected_context")
        if state_injected:
            await context.context_manager.inject_user_guidance(guidance=state_injected, step=current_step)

        capture = state.get("capture")
        if not capture:
            return {**state, "should_retry": True}

        start_time = time.time()
        
        # Prepare context
        smart_context = context.agent_state.get_smart_context()
        
        # Retrieve structured guidance from ContextManager
        guidance = context.context_manager.get_user_guidance()

        # Execute Planner
        plan = await context.planner.plan_step(
            state=context.agent_state,
            use_xml=context.use_xml,
            capture=capture,
            reasoner=context.reasoner,
            elements=state.get("elements"),
            additional_context=smart_context,
            guidance=guidance, 
        )
        
        duration = time.time() - start_time
        context.metrics.record(operation="analysis", duration=duration)
        
        if plan.metrics:
            context.metrics.record_tokens(
                prompt=int(plan.metrics.get("prompt_tokens", 0)),
                completion=int(plan.metrics.get("completion_tokens", 0)),
                cached=int(plan.metrics.get("cached_tokens", 0)),
            )
        
        if plan.step:
            context.ux.render_fallback(
                reasoning=plan.reason or "No reasoning",
                action=plan.step.action.to_description(),
                step_number=context.agent_state.step_count + 1,
            )

        return {
            **state,
            "plan": plan,
            "planned_step": plan.step,
            "is_complete": plan.is_complete,
            "completion_reason": plan.reason if plan.is_complete else state.get("completion_reason"),
            "should_retry": plan.should_retry,
            "analysis_duration": duration,
            "injected_context": None, # Consumed
        }

    async def execute_node(state: IntentGraphState) -> IntentGraphState:
        """
        Execute the planned action.
        """
        if context.is_cancelled:
            return {**state, "is_complete": True}

        injected = await _check_signal()
        if injected:
            context.telemetry.info("Context injected during execution phase. Triggering re-plan.")
            # Inject and route back
            await context.context_manager.inject_user_guidance(guidance=injected, step=context.agent_state.step_count)
            return {
                **state,
                "should_retry": True, 
                "planned_step": None, 
            }

        step = state.get("planned_step")
        capture = state.get("capture")
        if not step or not capture:
            return state

        start_time = time.time()
        
        # Resolve References
        resolved_action = await context.resolution.resolve(action=step.action)
        step = step.model_copy(update={"action": resolved_action})
        
        # Physical Execution
        size = await context.device.get_screen_size()
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
                device_result = await context.device.tap(x=x, y=y)
                
            elif action.action_type == ActionType.TYPE:
                if action.bounds:
                    coords = converter.center_to_pixels(bounds=action.bounds)
                    x, y = coords
                    await context.device.tap(x=x, y=y)
                device_result = await context.device.type_text(text=action.text or "")
                
            elif "swipe" in action.action_type.value:
                # Handle swipes
                direction = "up"
                if "_" in action.action_type.value:
                    direction = action.action_type.value.split("_")[1]
                
                bounds = action.bounds or Bounds(x=200, y=200, width=600, height=600)
                coords = converter.swipe_coordinates(bounds=bounds, direction=direction)
                x1, y1, x2, y2 = coords
                device_result = await context.device.swipe(x1=x1, y1=y1, x2=x2, y2=y2)
                
            elif action.action_type == ActionType.SCROLL:
                # Default scroll
                x1, y1, x2, y2 = size[0]//2, size[1]//2 + 300, size[0]//2, size[1]//2 - 300
                coords = (x1, y1, x2, y2)
                device_result = await context.device.swipe(x1=x1, y1=y1, x2=x2, y2=y2)
                
            elif action.action_type in (ActionType.BACK, ActionType.HOME):
                if action.action_type == ActionType.BACK:
                    device_result = await context.device.back()
                else:
                    device_result = await context.device.home()
            
            elif action.action_type == ActionType.WAIT:
                await asyncio.sleep((action.wait_duration or 1000) / 1000)
                device_result = ActionResult(success=True, duration=action.wait_duration or 1000)
            
            # Tracing
            if coords:
                # Use dynamic package name from state for correct folder
                pkg = "unknown"
                screen_state = state.get("screen_state")
                if screen_state and screen_state.activity:
                    pkg = screen_state.activity
                elif context.package_name:
                    pkg = context.package_name
                
                # Execute in background (fire and forget)
                asyncio.create_task(
                    asyncio.to_thread(
                        context.trace.save,
                        image_data=capture.image,
                        action=action,
                        coords=coords,
                        package_name=pkg,
                        session_id=context.workflow_id,
                        step_number=context.agent_state.step_count
                    )
                )
                
        except Exception as exception:
            device_result = ActionResult(success=False, duration=0, error=str(exception))

        duration = time.time() - start_time
        context.metrics.record(operation="action", duration=duration)
        
        step_result = StepResult(
            step=step,
            pre_hash=state.get("screen_state").visual_hash if state.get("screen_state") else "0",
            post_hash="0", 
            success=device_result.success,
            duration=int(duration * 1000),
            error=device_result.error,
            screen_changed=True, 
        )

        return {
            **state,
            "step_result": step_result,
            "execution_duration": duration,
        }

    async def record_node(state: IntentGraphState) -> IntentGraphState:
        """
        Record the execution result.
        """
        if context.is_cancelled:
            return {**state, "is_complete": True}

        step_result = state.get("step_result")
        if not step_result:
            return state

        context.agent_state.record_step(result=step_result)
        context.history.save_step(result=step_result, intent=context.intent)
        
        await context.memory.store_experience(
            visual_hash=step_result.pre_hash,
            action=step_result.step.action,
            success=step_result.success,
        )
        
        context.audit.log_step(
            plan=state.get("plan"),
            state=state.get("screen_state"),
            result=ActionResult(success=step_result.success, duration=step_result.duration),
            is_new_screen=state.get("is_new_screen"),
            is_stuck=context.agent_state.is_stuck,
            step_count=context.agent_state.step_count,
            analysis_duration=state.get("analysis_duration", 0),
            grounding_duration=state.get("grounding_duration", 0),
            hierarchy_duration=0,
            execution_duration=state.get("execution_duration", 0),
            total_duration=state.get("grounding_duration", 0) + state.get("analysis_duration", 0) + state.get("execution_duration", 0), 
        )

        return state

    return {
        "ground": ground_node,
        "analyze": analyze_node,
        "execute": execute_node,
        "record": record_node,
    }
