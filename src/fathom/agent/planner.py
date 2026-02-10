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

        if not state.can_continue:
            if state.is_complete:
                return PlanResult(step=None, is_complete=True, reason="Intent completed")

            return PlanResult(
                step=None, is_complete=False, reason="Max steps or recovery exhausted"
            )

        # IMMEDIATE RECOVERY: If we are stuck, don't ask the LLM again.
        # This breaks the loop by forcing a navigation change (BACK/SCROLL/HOME).
        if state.is_stuck:
            logger.warning(msg="Agent is stuck in a loop. Forcing recovery action.")

            recovery_action = state.get_recovery_action()
            if recovery_action:
                return self.__build_plan_result(
                    capture=capture,
                    is_recovery=True,
                    action=recovery_action,
                    step_number=state.step_count,
                )

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
            context=full_context,
            failures=context.get("relevant_failures", []),  # type: ignore[arg-type]
        )

        completion = reasoner.analyze_completion(
            analysis=analysis, screen_description=capture.activity
        )

        if completion.is_complete:
            state.mark_complete(reason=completion.evidence)
            return PlanResult(
                step=None,
                is_complete=True,
                metrics=analysis.metrics,
                metadata=analysis.metadata,
                reason=completion.evidence,
                memories=analysis.memories,
            )

        action = self.__select_action(state=state, reasoner=reasoner, analysis=analysis)

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

        screen_hash = self.__compute_simple_hash(capture=capture)

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
            memories=memories,
            metrics=metrics or {},
            metadata=metadata or {},
            reason="Step planned" if not is_recovery else "Recovery step",
            is_valid_action=action.is_valid,
            validation_reasoning=action.validation_reason,
        )

    def __compute_simple_hash(self, capture: ScreenCapture) -> str:
        """
        Compute a simple hash of the screen capture
        """

        data = f"{capture.activity}:{len(capture.image)}".encode()
        return hashlib.md5(string=data, usedforsecurity=False).hexdigest()[:16]
