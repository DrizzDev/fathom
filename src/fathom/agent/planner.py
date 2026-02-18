from __future__ import annotations

import hashlib
from logging import getLogger
from typing import Any, Dict, Optional, Tuple

from fathom.agent.reasoner import Reasoner
from fathom.agent.state import AgentState
from fathom.schemas.actions import Action, Bounds
from fathom.schemas.results import AnalysisResult, PlanResult
from fathom.schemas.screens import ScreenCapture
from fathom.schemas.steps import Step
from fathom.tools.vision import VisionTool

logger = getLogger(name=__name__)


class CoordinateConverter:
    """
    Converts normalized coordinates to device pixels.
    """

    def __init__(self, screen_width: int, screen_height: int) -> None:
        self.__width = screen_width
        self.__height = screen_height

    def to_pixels(self, bounds: Bounds) -> Tuple[int, int, int, int]:
        """
        Convert a bounding box to pixel coordinates.
        """

        return bounds.to_pixels(screen_width=self.__width, screen_height=self.__height)

    def center_to_pixels(self, bounds: Bounds) -> Tuple[int, int]:
        """
        Convert a bounding box to its center pixel coordinates.
        """

        x, y, width, height = self.to_pixels(bounds=bounds)
        return x + width // 2, y + height // 2

    def swipe_coordinates(
        self,
        direction: str,
        bounds: Bounds,
    ) -> Tuple[int, int, int, int]:
        """
        Convert a bounding box and swipe direction to pixel coordinates.
        """

        x, y, width, height = self.to_pixels(bounds=bounds)
        center_x, center_y = x + width // 2, y + height // 2

        distance_x = int(width * 0.7)
        distance_y = int(height * 0.7)

        if direction == "up":
            return (
                center_x,
                center_y + distance_y // 2,
                center_x,
                center_y - distance_y // 2,
            )
        elif direction == "down":
            return (
                center_x,
                center_y - distance_y // 2,
                center_x,
                center_y + distance_y // 2,
            )
        elif direction == "left":
            return (
                center_x + distance_x // 2,
                center_y,
                center_x - distance_x // 2,
                center_y,
            )
        elif direction == "right":
            return (
                center_x - distance_x // 2,
                center_y,
                center_x + distance_x // 2,
                center_y,
            )
        return center_x, center_y, center_x, center_y


class StepPlanner:
    """
    Plans and prepares steps for execution.
    """

    def __init__(
        self,
        vision_tool: VisionTool,
        *,
        min_confidence: float = 0.4,
    ) -> None:
        self.__vision = vision_tool
        self.__min_confidence = min_confidence

    @property
    def vision_tool(self) -> VisionTool:
        """
        Returns the underlying vision tool.
        """

        return self.__vision

    async def plan_step(
        self,
        state: AgentState,
        reasoner: Reasoner,
        capture: ScreenCapture,
        *,
        use_xml: bool = False,
        elements: Optional[Dict[str, Any]] = None,
        additional_context: Optional[str] = None,
    ) -> PlanResult:
        """
        Plan the next step based on current state.
        """

        logger.debug(
            f"[H5] Entering planner plan_step | "
            f"step_count={state.step_count} is_complete={state.is_complete} "
            f"is_stuck={state.is_stuck} can_continue={state.can_continue}"
        )

        if not state.can_continue:
            if state.is_complete:
                return PlanResult(step=None, is_complete=True, reason="Intent completed")

            return PlanResult(
                step=None, is_complete=False, reason="Max steps or recovery exhausted"
            )

        if state.is_stuck:
            logger.warning(msg="Agent is stuck in a loop. Requesting recovery from model.")
            # Defer recording recovery attempt until after we check for completion flags.

        context = state.build_context()
        history_context = str(object=context.get("compact_history", "None"))

        full_context = history_context
        if additional_context:
            full_context = f"{additional_context}\n{history_context}"

        analysis = await self.__vision.analyze(
            use_xml=use_xml,
            capture=capture,
            elements=elements,
            intent=state.intent,
            is_stuck=state.is_stuck,
            last_action=(state.last_action_type.value if state.last_action_type else None),
            context=full_context,
            failures=context.get("relevant_failures", []),  # type: ignore[arg-type]
        )

        # Content Exhaustion Signal:
        # If model signals content exhaustion, reset loop detector to prevent
        # false stuck detection from repeated swipes on an unchanged screen.
        if analysis.content_exhausted:
            state.reset_loop_detector()
            state.mark_complete(reason="Content exhaustion signaled by model")
            return PlanResult(
                step=None,
                is_complete=True,
                reason="Model signaled content exhaustion (end of list/carousel).",
                metrics=analysis.metrics,
                metadata=analysis.metadata,
                memories=analysis.memories,
            )

        completion = reasoner.analyze_completion(
            analysis=analysis, screen_description=analysis.screen_description
        )

        action = self.__select_action(state=state, reasoner=reasoner, analysis=analysis)

        if completion.is_complete:
            state.mark_complete(reason=completion.evidence)

            # If there's a valid physical action, we should execute it before finishing
            step = None
            if action.action_type not in ("complete", "unknown", "wait"):
                step = self.__build_step(
                    action=action, step_number=state.step_count, capture=capture
                )

            return PlanResult(
                step=step,
                is_complete=True,
                metrics=analysis.metrics,
                metadata=analysis.metadata,
                reason=completion.evidence,
                memories=analysis.memories,
            )

        # If we are NOT complete and still stuck, NOW we record the attempt.
        if state.is_stuck:
            state.record_recovery_attempt()

        # Optimization: Check if this EXACT action just failed on this screen hash
        if state.should_avoid_action(action=action):
            logger.warning(msg=f"Avoiding recently failed action: {action.to_description()}")
            return PlanResult(
                step=None,
                is_complete=False,
                should_retry=True,
                metrics=analysis.metrics,
                metadata=analysis.metadata,
                memories=analysis.memories,
                reason="Action recently failed on this screen",
            )

        if not reasoner.should_accept_action(
            action=action, has_failed_before=state.should_avoid_action(action=action)
        ):
            return PlanResult(
                step=None,
                is_complete=False,
                should_retry=True,
                metrics=analysis.metrics,
                metadata=analysis.metadata,
                memories=analysis.memories,
                reason=f"Action rejected: {action.rationale}",
            )

        return self.__build_plan_result(
            action=action,
            capture=capture,
            metrics=analysis.metrics,
            metadata=analysis.metadata,
            memories=analysis.memories,
            step_number=state.step_count,
        )

    def __select_action(
        self,
        state: AgentState,
        reasoner: Reasoner,
        analysis: AnalysisResult,
    ) -> Action:
        """
        Return the best action from the analysis result.
        """

        context = state.build_context()
        failures_raw = context.get("relevant_failures", [])
        failures = failures_raw if isinstance(failures_raw, list) else []

        return reasoner.select_best_action(
            primary=analysis.action,
            alternatives=analysis.alternatives,
            failed_actions={str(object=failure) for failure in failures},
        )

    def __build_plan_result(
        self,
        action: Action,
        step_number: int,
        capture: ScreenCapture,
        *,
        memories: int = 0,
        is_recovery: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, float]] = None,
    ) -> PlanResult:
        """
        Return a PlanResult with the given action and metadata.
        """

        step = self.__build_step(
            action=action, step_number=step_number, capture=capture, is_recovery=is_recovery
        )

        return PlanResult(
            step=step,
            is_complete=False,
            memories=memories,
            metrics=metrics or {},
            metadata=metadata or {},
            reason="Step planned" if not is_recovery else "Recovery step",
            is_valid_action=action.is_valid,
            validation_reasoning=action.validation_reason,
        )

    def __build_step(
        self,
        action: Action,
        step_number: int,
        capture: ScreenCapture,
        is_recovery: bool = False,
    ) -> Step:
        """
        Helper to construct a Step object.
        """

        screen_hash = self.__compute_simple_hash(capture=capture)

        return Step(
            action=action,
            screen_hash=screen_hash,
            step_number=step_number,
            is_conditional=is_recovery,
            condition="recovery" if is_recovery else None,
        )

    def __compute_simple_hash(self, capture: ScreenCapture) -> str:
        """
        Compute a simple hash of the screen capture
        """

        data = f"{capture.activity}:{len(capture.image)}".encode()
        return hashlib.md5(string=data, usedforsecurity=False).hexdigest()[:16]
