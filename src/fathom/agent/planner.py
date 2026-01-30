"""Step planning with coordinate conversion and execution preparation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from fathom.agent.reasoner import Reasoner
from fathom.agent.state import AgentState
from fathom.constants import ActionType
from fathom.schemas.actions import Action, BoundingBox
from fathom.schemas.screens import ScreenCapture
from fathom.schemas.steps import Step
from fathom.tools.vision import AnalysisResult, VisionTool


@dataclass(frozen=True)
class PlanResult:
    """Result of step planning."""

    step: Optional[Step]
    is_complete: bool
    reason: str
    should_retry: bool = False


class CoordinateConverter:
    """Converts normalized coordinates to device pixels.

    Handles the translation from 0-1000 normalized scale
    to actual device pixel coordinates.
    """

    def __init__(self, screen_width: int, screen_height: int) -> None:
        """Initialize converter.

        Args:
            screen_width: Device screen width in pixels.
            screen_height: Device screen height in pixels.
        """
        self.__width = screen_width
        self.__height = screen_height

    def to_pixels(self, bbox: BoundingBox) -> Tuple[int, int, int, int]:
        """Convert bounding box to pixel coordinates.

        Args:
            bbox: Normalized bounding box (0-1000 scale).

        Returns:
            Tuple of (x, y, width, height) in pixels.
        """
        return bbox.to_pixels(self.__width, self.__height)

    def center_to_pixels(self, bbox: BoundingBox) -> Tuple[int, int]:
        """Get center point in pixel coordinates.

        Args:
            bbox: Normalized bounding box.

        Returns:
            Tuple of (x, y) center point in pixels.
        """
        x, y, w, h = self.to_pixels(bbox)
        return x + w // 2, y + h // 2

    def swipe_coordinates(
        self,
        bbox: BoundingBox,
        direction: str,
    ) -> Tuple[int, int, int, int]:
        """Calculate swipe start and end coordinates.

        Args:
            bbox: Target area for swipe.
            direction: One of 'up', 'down', 'left', 'right'.

        Returns:
            Tuple of (x1, y1, x2, y2) for swipe.
        """
        x, y, w, h = self.to_pixels(bbox)
        cx, cy = x + w // 2, y + h // 2

        distance_x = int(w * 0.7)
        distance_y = int(h * 0.7)

        if direction == "up":
            return cx, cy + distance_y // 2, cx, cy - distance_y // 2
        elif direction == "down":
            return cx, cy - distance_y // 2, cx, cy + distance_y // 2
        elif direction == "left":
            return cx + distance_x // 2, cy, cx - distance_x // 2, cy
        elif direction == "right":
            return cx - distance_x // 2, cy, cx + distance_x // 2, cy
        else:
            return cx, cy, cx, cy


class StepPlanner:
    """Plans and prepares steps for execution.

    Orchestrates:
    - Vision analysis for action recommendation
    - Coordinate conversion for device interaction
    - Step construction with proper metadata
    - Recovery handling when stuck

    The planner is stateless - all state is managed by AgentState.
    """

    def __init__(
        self,
        vision_tool: VisionTool,
        *,
        min_confidence: float = 0.4,
    ) -> None:
        """Initialize planner.

        Args:
            vision_tool: Vision tool for screen analysis.
            min_confidence: Minimum confidence for action acceptance.
        """
        self.__vision = vision_tool
        self.__min_confidence = min_confidence

    async def plan_step(
        self,
        state: AgentState,
        capture: ScreenCapture,
        reasoner: Reasoner,
    ) -> PlanResult:
        """Plan the next step based on current state.

        Args:
            state: Current agent state.
            capture: Current screen capture.
            reasoner: Reasoner for completion detection.

        Returns:
            PlanResult with step to execute or completion info.
        """
        if not state.can_continue:
            if state.is_complete:
                return PlanResult(
                    step=None,
                    is_complete=True,
                    reason="Intent completed",
                )
            if state.is_stuck:
                recovery = state.get_recovery_action()
                if recovery:
                    return self.__build_plan_result(
                        recovery,
                        capture,
                        state.step_count,
                        is_recovery=True,
                    )
                return PlanResult(
                    step=None,
                    is_complete=False,
                    reason="Stuck in loop, recovery exhausted",
                )
            return PlanResult(
                step=None,
                is_complete=False,
                reason=f"Max steps ({state.step_count}) exceeded",
            )

        context = state.build_context()
        analysis = await self.__vision.analyze(
            screen=capture.image,
            intent=state.intent,
            context=context.get("recent_actions", []),  # type: ignore[arg-type]
            failures=context.get("recent_failures", []),  # type: ignore[arg-type]
        )

        completion = reasoner.analyze_completion(
            analysis,
            screen_description=capture.activity,
        )

        if completion.is_complete:
            state.mark_complete(completion.evidence)
            return PlanResult(
                step=None,
                is_complete=True,
                reason=completion.evidence,
            )

        action = self.__select_action(analysis, state, reasoner)

        if not reasoner.should_accept_action(
            action,
            has_failed_before=state.should_avoid_action(action),
        ):
            return PlanResult(
                step=None,
                is_complete=False,
                reason=f"Action rejected: confidence {action.confidence:.2f}",
                should_retry=True,
            )

        return self.__build_plan_result(action, capture, state.step_count)

    def __select_action(
        self,
        analysis: AnalysisResult,
        state: AgentState,
        reasoner: Reasoner,
    ) -> Action:
        """Select best action from analysis result."""
        context = state.build_context()
        failures_raw = context.get("recent_failures", [])
        failures = failures_raw if isinstance(failures_raw, list) else []
        failed_descs = {str(f) for f in failures}

        return reasoner.select_best_action(
            analysis.action,
            analysis.alternatives,
            failed_actions=failed_descs,
        )

    def __build_plan_result(
        self,
        action: Action,
        capture: ScreenCapture,
        step_number: int,
        *,
        is_recovery: bool = False,
    ) -> PlanResult:
        """Build PlanResult from action."""
        screen_hash = self.__compute_simple_hash(capture)

        step = Step(
            action=action,
            screen_hash=screen_hash,
            step_number=step_number,
            is_conditional=is_recovery,
            condition="recovery" if is_recovery else None,
        )

        return PlanResult(
            step=step,
            is_complete=False,
            reason="Step planned" if not is_recovery else "Recovery step",
        )

    def __compute_simple_hash(self, capture: ScreenCapture) -> str:
        """Compute simple hash for screen capture."""
        import hashlib

        data = f"{capture.activity}:{len(capture.image)}".encode()
        return hashlib.md5(data).hexdigest()[:16]  # nosec

    def prepare_execution(
        self,
        step: Step,
        screen_width: int,
        screen_height: int,
    ) -> dict[str, object]:
        """Prepare step for device execution.

        Converts normalized coordinates to pixels and builds
        the execution request for the device tool.

        Args:
            step: Step to prepare.
            screen_width: Device screen width.
            screen_height: Device screen height.

        Returns:
            Execution request dict for device tool.
        """
        action = step.action
        converter = CoordinateConverter(screen_width, screen_height)

        if action.action_type == ActionType.TAP:
            if action.bbox:
                x, y = converter.center_to_pixels(action.bbox)
            else:
                x, y = screen_width // 2, screen_height // 2
            return {"action": "tap", "x": x, "y": y}

        elif action.action_type == ActionType.TYPE:
            return {"action": "type", "text": action.text or ""}

        elif action.action_type == ActionType.SWIPE:
            if action.bbox:
                x1, y1, x2, y2 = converter.swipe_coordinates(action.bbox, "up")
            else:
                cx, cy = screen_width // 2, screen_height // 2
                x1, y1 = cx, cy + 300
                x2, y2 = cx, cy - 300
            return {"action": "swipe", "x1": x1, "y1": y1, "x2": x2, "y2": y2}

        elif action.action_type == ActionType.SCROLL:
            cx, cy = screen_width // 2, screen_height // 2
            return {
                "action": "swipe",
                "x1": cx,
                "y1": cy + 400,
                "x2": cx,
                "y2": cy - 400,
                "duration": 500,
            }

        elif action.action_type == ActionType.LONG_PRESS:
            if action.bbox:
                x, y = converter.center_to_pixels(action.bbox)
            else:
                x, y = screen_width // 2, screen_height // 2
            return {
                "action": "swipe",
                "x1": x,
                "y1": y,
                "x2": x,
                "y2": y,
                "duration": 1000,
            }

        elif action.action_type == ActionType.BACK:
            return {"action": "back"}

        elif action.action_type == ActionType.HOME:
            return {"action": "home"}

        elif action.action_type == ActionType.WAIT:
            return {"action": "wait", "duration": 1000}

        elif action.action_type == ActionType.COMPLETE:
            return {"action": "complete"}

        else:
            return {"action": "unknown"}
