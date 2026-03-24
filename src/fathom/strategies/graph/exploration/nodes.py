from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from typing import Any, Callable, Dict, cast

from fathom.constants import ActionType
from fathom.constants.execution import VISUAL_HASH_LENGTH
from fathom.constants.state import CommonStateKey as CKey
from fathom.constants.state import ExplorationStateKey as EKey
from fathom.schemas.actions import Action
from fathom.schemas.screens import ScreenCapture, ScreenState
from fathom.schemas.steps import Step, StepResult
from fathom.strategies.graph.context import GraphContext
from fathom.strategies.graph.exploration.state import (
    ExplorationGraphState,
    get_action,
    get_capture,
    get_screen_state,
    get_step_result,
    is_content_exhausted,
)
from fathom.utils.wait import stability_wait

logger = logging.getLogger(__name__)


class ExplorationNodeProvider:
    """
    Provides LangGraph nodes for application exploration.
    Encapsulates dependencies and shared private logic.
    """

    def __init__(self, context: GraphContext) -> None:
        """
        Initialize provider with shared context.
        """

        self.__context = context

    async def ground(self, state: ExplorationGraphState) -> ExplorationGraphState:
        """
        Capture screen and update state.
        """

        if self.__context.is_cancelled:
            result = cast("dict[str, Any]", dict(state))
            result[CKey.IS_COMPLETE] = True
            result[CKey.COMPLETION_REASON] = "Cancelled"
            return cast("ExplorationGraphState", result)

        start_time = time.time()

        try:
            screenshot_result, dimensions_result, activity_result = await asyncio.gather(
                self.__context.device.capture_screen(),
                self.__context.device.get_dimensions(),
                self.__context.device.get_current_package(),
                return_exceptions=True,
            )

            if isinstance(screenshot_result, BaseException):
                raise screenshot_result
            screenshot_bytes: bytes = screenshot_result

            if isinstance(dimensions_result, BaseException):
                raise dimensions_result
            width, height = dimensions_result

            if isinstance(activity_result, BaseException):
                activity = "unknown"
                await self.__context.telemetry.warning(
                    "Failed to get current package", error=str(activity_result)
                )
            else:
                activity = str(activity_result)

            storage_id = await self.__context.storage.save(
                data=screenshot_bytes,
                metadata={
                    "type": "screenshot",
                    "category": "screenshot",
                    "package_name": activity,
                    "timestamp": time.time(),
                    "session_id": self.__context.workflow_id,
                    "filename": f"{int(time.time() * 1000)}__{activity}.png",
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

            is_new = self.__context.agent_state.update_screen(screen=screen_state)

            result = cast("dict[str, Any]", dict(state))
            result[CKey.CAPTURE] = screen
            result[CKey.SCREEN_STATE] = screen_state
            result[CKey.IS_NEW_SCREEN] = is_new
            result[CKey.GROUNDING_DURATION] = time.time() - start_time
            result[CKey.STEP_RESULT] = None
            result[EKey.ACTION] = None
            return cast("ExplorationGraphState", result)

        except Exception as exception:
            logger.error(f"Exploration grounding failed: {exception}")
            result = cast("dict[str, Any]", dict(state))
            result[CKey.IS_COMPLETE] = True
            result[CKey.COMPLETION_REASON] = "Capture failed"
            return cast("ExplorationGraphState", result)

    async def scan(self, state: ExplorationGraphState) -> ExplorationGraphState:
        """
        Scan the screen using Vision Service to find next action.
        """

        if self.__context.is_cancelled:
            result = cast("dict[str, Any]", {})
            result[CKey.IS_COMPLETE] = True
            return cast("ExplorationGraphState", result)

        capture = get_capture(state)
        if not capture:
            result = cast("dict[str, Any]", dict(state))
            result[EKey.CONTENT_EXHAUSTED] = True
            return cast("ExplorationGraphState", result)

        start = time.time()
        width, height = await self.__context.device.get_dimensions()

        analysis = await self.__context.vision.analyze(
            capture=capture,
            tracking_note=None,
            screen_width=width,
            screen_height=height,
            context_manager=self.__context.context_manager,
            intent="Explore this app. Find a unique interactive element.",
        )

        if (
            analysis.is_goal_complete
            or not analysis.action
            or analysis.action.action_type == ActionType.COMPLETE
        ):
            exhausted = True
        else:
            exhausted = False

        result = cast("dict[str, Any]", dict(state))
        result[CKey.ANALYSIS] = analysis
        result[EKey.CONTENT_EXHAUSTED] = exhausted
        result[CKey.ANALYSIS_DURATION] = time.time() - start
        result[EKey.ACTION] = analysis.action if not exhausted else None

        return cast("ExplorationGraphState", result)

    async def execute(self, state: ExplorationGraphState) -> ExplorationGraphState:
        """
        Execute the action via ActionExecutor.
        """

        if self.__context.is_cancelled:
            result = cast("dict[str, Any]", {})
            result[CKey.IS_COMPLETE] = True
            return cast("ExplorationGraphState", result)

        action = get_action(state)
        capture = get_capture(state)

        if not action or not capture:
            return state

        start_time = time.time()

        # Step construction for ActionExecutor
        step = Step(
            action=action,
            screen_hash="0",
            step_number=self.__context.agent_state.step_count,
        )

        screen_state = get_screen_state(state)

        if screen_state and screen_state.activity:
            package_name = screen_state.activity
        else:
            package_name = "unknown"

        # Delegate to ActionExecutor for consistent retries and tracing
        execution_result = await self.__context.action_executor.act(
            step=step,
            pre_capture=capture,
            package_name=package_name,
            session_id=self.__context.workflow_id,
        )

        # Post-action stability wait with hard cap for consistency.
        await stability_wait(self.__context.configuration)

        duration = time.time() - start_time

        step_result = StepResult(
            step=step,
            post_hash="0",
            screen_changed=True,
            duration=int(duration * 1000),
            success=execution_result.success,
            pre_hash=screen_state.visual_hash if screen_state else "0",
        )

        result = cast("dict[str, Any]", dict(state))
        result[CKey.STEP_RESULT] = step_result
        result[CKey.EXECUTION_DURATION] = duration

        return cast("ExplorationGraphState", result)

    async def record(self, state: ExplorationGraphState) -> ExplorationGraphState:
        """
        Record result and update queues.
        """

        if self.__context.is_cancelled:
            result = cast("dict[str, Any]", {})
            result[CKey.IS_COMPLETE] = True
            return cast("ExplorationGraphState", result)

        step_result = get_step_result(state)
        if not step_result:
            return state

        screen_state = get_screen_state(state)
        if isinstance(screen_state, ScreenState):
            self.__context.exploration_graph.add_screen(screen_state)

        if step_result.success and step_result.pre_hash and step_result.step.action:
            self.__context.exploration_graph.record_transition(
                destination="0",
                origin=step_result.pre_hash,
                action=step_result.step.action.to_description(),
            )

        self.__context.agent_state.record_step(result=step_result)
        activity = screen_state.activity if isinstance(screen_state, ScreenState) else "unknown"
        await self.__context.history.save_step(
            result=step_result,
            intent="exploration",
            activity=activity,
        )

        if self.__context.agent_state.step_count >= self.__context.max_steps:
            result = cast("dict[str, Any]", dict(state))
            result[CKey.IS_COMPLETE] = True
            result[CKey.COMPLETION_REASON] = "Max steps"
            return cast("ExplorationGraphState", result)

        return state

    async def navigate(self, state: ExplorationGraphState) -> ExplorationGraphState:
        """
        Navigate to target screen using BFS path.
        Executes actions from pending_nav queue to reach unexplored screens.
        """

        if self.__context.is_cancelled:
            result = cast("dict[str, Any]", dict(state))
            result[CKey.IS_COMPLETE] = True
            return cast("ExplorationGraphState", result)

        pending_nav = state.get(EKey.PENDING_NAV, [])

        # If no pending navigation, we're done with this path
        if not pending_nav:
            result = cast("dict[str, Any]", dict(state))
            result[EKey.BFS_PHASE] = "scan"
            return cast("ExplorationGraphState", result)

        # Pop the next action to execute
        action_dict = pending_nav[0]
        remaining_nav = pending_nav[1:]

        action = Action(**action_dict)

        # Execute the navigation action
        step = Step(
            action=action,
            screen_hash="0",
            step_number=self.__context.agent_state.step_count,
        )

        capture = get_capture(state)
        if not capture:
            result = cast("dict[str, Any]", dict(state))
            result[EKey.BFS_PHASE] = "scan"
            return cast("ExplorationGraphState", result)

        await self.__context.action_executor.act(
            step=step,
            pre_capture=capture,
            session_id=self.__context.workflow_id,
            package_name=self.__context.package_name,
        )

        # Wait for stability with hard cap for consistency.
        await stability_wait(self.__context.configuration)

        # Update state with remaining navigation
        result = cast("dict[str, Any]", dict(state))
        result[EKey.PENDING_NAV] = remaining_nav
        result[EKey.BFS_PHASE] = "scan" if not remaining_nav else "navigate"

        return cast("ExplorationGraphState", result)

    async def bfs_route(self, state: ExplorationGraphState) -> ExplorationGraphState:
        """
        Decide next BFS phase based on exploration state.
        Routes between scanning current screen, returning to parent, or advancing to new screen.
        """

        if self.__context.is_cancelled:
            result = cast("dict[str, Any]", dict(state))
            result[CKey.IS_COMPLETE] = True
            return cast("ExplorationGraphState", result)

        # Get BFS state
        visited_hashes = state.get(EKey.VISITED_HASHES, set())

        # If current screen is exhausted, try to find next unexplored screen
        if is_content_exhausted(state):
            # Look for unexplored screens in the graph
            unexplored = []
            for hash_val, node in self.__context.exploration_graph.nodes.items():
                if hash_val not in visited_hashes and node.should_explore():
                    unexplored.append(hash_val)

            if unexplored:
                # Pick the first unexplored screen
                target_hash = unexplored[0]

                # Find path to target (simplified - just mark as visited)
                # In a full implementation, this would use BFS to find the shortest path
                visited_hashes.add(target_hash)

                result = cast("dict[str, Any]", dict(state))
                result[EKey.SCANNING_HASH] = target_hash
                result[EKey.VISITED_HASHES] = visited_hashes
                result[EKey.BFS_PHASE] = "navigate"
                result[EKey.PENDING_NAV] = []  # Would contain path actions in full implementation
                return cast("ExplorationGraphState", result)
            else:
                # No more screens to explore
                result = cast("dict[str, Any]", dict(state))
                result[CKey.IS_COMPLETE] = True
                result[CKey.COMPLETION_REASON] = "All screens explored"
                return cast("ExplorationGraphState", result)

        # Continue scanning current screen
        result = cast("dict[str, Any]", dict(state))
        result[EKey.BFS_PHASE] = "scan"
        return cast("ExplorationGraphState", result)


class ExplorationGraphFactory:
    """
    Factory for building the Exploration Node functions.
    """

    @staticmethod
    def build(context: GraphContext) -> Dict[str, Callable[..., Any]]:
        """
        Builds the node functions for the exploration graph.
        """

        from fathom.constants.graph import NodeName

        provider = ExplorationNodeProvider(context=context)

        return {
            NodeName.SCAN: provider.scan,
            NodeName.RECORD: provider.record,
            NodeName.GROUND: provider.ground,
            NodeName.EXECUTE: provider.execute,
            NodeName.NAVIGATE: provider.navigate,
            NodeName.BFS_ROUTE: provider.bfs_route,
        }
