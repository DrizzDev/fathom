"""
Graph nodes for exploration execution.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict

from fathom.constants import ActionType
from fathom.schemas.screens import ScreenCapture, ScreenState
from fathom.schemas.steps import Step, StepResult
from fathom.strategies.graph.context import GraphContext
from fathom.strategies.graph.exploration.state import ExplorationGraphState
from fathom.utils.coordinates import CoordinateConverter


def build_exploration_nodes(context: GraphContext) -> Dict[str, Callable[..., Any]]:
    """
    Builds the node functions for the exploration graph.
    """

    async def ground_node(state: ExplorationGraphState) -> ExplorationGraphState:
        """Capture screen and update state."""
        if context.is_cancelled:
            return {**state, "is_complete": True}

        start_time = time.time()

        try:
            screenshot_bytes = await context.device.capture_screen()
            width, height = await context.device.get_screen_size()

            try:
                activity = await context.device.get_current_package()
            except Exception:
                activity = "unknown"

            storage_id = await context.storage.save(
                data=screenshot_bytes,
                metadata={
                    "type": "screenshot",
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

            # Compute Hash
            import hashlib

            from fathom.constants.execution import VISUAL_HASH_LENGTH

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

            is_new = context.agent_state.update_screen(screen=screen_state)

            return {
                **state,
                "capture": screen,
                "screen_state": screen_state,
                "is_new_screen": is_new,
                "grounding_duration": time.time() - start_time,
                "step_result": None,
                "action": None,
            }

        except Exception:
            return {**state, "is_complete": True, "completion_reason": "Capture failed"}

    async def scan_node(state: ExplorationGraphState) -> ExplorationGraphState:
        """
        Scan the screen using Vision Service to find next action.
        """
        if context.is_cancelled:
            return {**state, "is_complete": True}

        capture = state.get("capture")
        if not capture:
            return {**state, "content_exhausted": True}

        start = time.time()

        # Use Vision Service to find *unexplored* elements.
        # Ideally we pass context about what we've already done on this screen.
        # KnowledgeGraph (context.knowledge) stores this?
        # GraphContext doesn't expose KG lookup directly?
        # We'll use agent_state history for now.

        analysis = await context.vision.analyze(
            intent="Explore this app. Find a unique interactive element that hasn't been clicked yet.",
            capture=capture,
            context=context.agent_state.build_context().get("compact_history"),
        )

        # If no action found or action is "complete", mark exhausted
        exhausted = False
        if (
            analysis.is_goal_complete
            or not analysis.action
            or analysis.action.action_type == ActionType.COMPLETE
        ):
            exhausted = True

        return {
            **state,
            "action": analysis.action if not exhausted else None,
            "analysis": analysis,
            "content_exhausted": exhausted,
            "analysis_duration": time.time() - start,
        }

    async def execute_node(state: ExplorationGraphState) -> ExplorationGraphState:
        """Execute the action."""
        if context.is_cancelled:
            return {**state, "is_complete": True}

        action = state.get("action")
        if not action:
            return state

        start = time.time()
        size = await context.device.get_screen_size()
        converter = CoordinateConverter(screen_width=size[0], screen_height=size[1])

        # Execute logic (simplified tap/scroll)
        success = False
        try:
            if action.action_type == ActionType.TAP:
                if action.bounds:
                    x, y = converter.center_to_pixels(bounds=action.bounds)
                    await context.device.tap(x=x, y=y)
                    success = True
            elif action.action_type == ActionType.SCROLL:
                x1, y1, x2, y2 = size[0] // 2, size[1] // 2 + 300, size[0] // 2, size[1] // 2 - 300
                await context.device.swipe(x1=x1, y1=y1, x2=x2, y2=y2)
                success = True
            elif action.action_type == ActionType.BACK:
                await context.device.back()
                success = True
        except Exception:
            success = False

        duration = time.time() - start

        step = Step(
            action=action,
            step_number=context.agent_state.step_count,
            screen_hash=state.get("screen_state").visual_hash if state.get("screen_state") else "0",
        )

        step_result = StepResult(
            step=step,
            success=success,
            duration=int(duration * 1000),
            screen_changed=True,
            pre_hash=step.screen_hash,
            post_hash="0",
        )

        return {
            **state,
            "step_result": step_result,
            "execution_duration": duration,
        }

    async def record_node(state: ExplorationGraphState) -> ExplorationGraphState:
        """Record result and update queues."""
        if context.is_cancelled:
            return {**state, "is_complete": True}

        step_result = state.get("step_result")
        if not step_result:
            return state

        # Update Exploration Graph
        if state.get("screen_state"):
            context.exploration_graph.add_screen(state.get("screen_state"))

        if step_result.success and step_result.pre_hash and step_result.step.action:
            context.exploration_graph.record_transition(
                origin=step_result.pre_hash,
                destination="0",  # Post hash not available yet?
                action=step_result.step.action.to_description(),
            )

        # Record
        context.agent_state.record_step(result=step_result)
        context.history.save_step(result=step_result, intent="exploration")

        # Determine next phase (Simplified BFS logic)
        # In a real impl, we'd check if screen changed, add to queue, etc.
        # For now, we rely on the loop.

        # Check max steps
        if context.agent_state.step_count >= context.max_steps:
            return {**state, "is_complete": True, "completion_reason": "Max steps"}

        return state

    async def navigate_node(state: ExplorationGraphState) -> ExplorationGraphState:
        """Navigate to target screen (placeholder for BFS navigation)."""
        # For now, just reset to SCAN phase
        return {**state, "bfs_phase": "scan"}

    async def bfs_route_node(state: ExplorationGraphState) -> ExplorationGraphState:
        """Decide next phase."""
        # Simple logic: If exhausted, maybe try navigation or end.
        return state

    return {
        "ground": ground_node,
        "scan": scan_node,
        "execute": execute_node,
        "record": record_node,
        "navigate": navigate_node,
        "bfs_route": bfs_route_node,
    }
