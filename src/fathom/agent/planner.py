from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Optional, Tuple

from fathom.agent.reasoner import Reasoner
from fathom.agent.state import AgentState
from fathom.schemas.actions import Action, BoundingBox
from fathom.schemas.screens import ScreenCapture
from fathom.schemas.steps import Step
from fathom.tools.vision import AnalysisResult, VisionTool


@dataclass(frozen=True)
class PlanResult:
    """
    Result of step planning.
    """

    reason: str
    is_complete: bool
    step: Optional[Step]
    memories: int = 0
    metrics: dict[str, float] = field(default_factory=dict)
    should_retry: bool = False


class CoordinateConverter:
    """
    Converts normalized coordinates to device pixels.
    """

    def __init__(self, screen_width: int, screen_height: int) -> None:
        self.__width = screen_width
        self.__height = screen_height

    def to_pixels(self, bbox: BoundingBox) -> Tuple[int, int, int, int]:
        return bbox.to_pixels(self.__width, self.__height)

    def center_to_pixels(self, bbox: BoundingBox) -> Tuple[int, int]:
        x, y, w, h = self.to_pixels(bbox)
        return x + w // 2, y + h // 2

    def swipe_coordinates(
        self,
        bbox: BoundingBox,
        direction: str,
    ) -> Tuple[int, int, int, int]:
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
        return cx, cy, cx, cy


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

    async def plan_step(
        self,
        state: AgentState,
        reasoner: Reasoner,
        capture: ScreenCapture,
        *,
        use_xml: bool = False,
    ) -> PlanResult:
        """
        Plan the next step based on current state.
        """
        if not state.can_continue:
            if state.is_complete:
                return PlanResult(step=None, is_complete=True, reason="Intent completed")

            if state.is_stuck:
                recovery = state.get_recovery_action()
                if recovery:
                    return self.__build_plan_result(
                        action=recovery,
                        capture=capture,
                        is_recovery=True,
                        step_number=state.step_count,
                    )
                return PlanResult(step=None, is_complete=False, reason="Stuck in loop")
            return PlanResult(step=None, is_complete=False, reason="Max steps exceeded")

        context = state.build_context()
        analysis = await self.__vision.analyze(
            use_xml=use_xml,
            capture=capture,
            intent=state.intent,
            context=context.get("recent_actions", []),  # type: ignore[arg-type]
            failures=context.get("recent_failures", []),  # type: ignore[arg-type]
        )

        completion = reasoner.analyze_completion(analysis, screen_description=capture.activity)

        if completion.is_complete:
            state.mark_complete(completion.evidence)
            return PlanResult(
                step=None,
                is_complete=True,
                reason=completion.evidence,
                memories=analysis.memories,
                metrics=analysis.metrics,
            )

        action = self.__select_action(state=state, reasoner=reasoner, analysis=analysis)

        if not reasoner.should_accept_action(
            action, has_failed_before=state.should_avoid_action(action)
        ):
            return PlanResult(
                step=None,
                is_complete=False,
                should_retry=True,
                reason=f"Action rejected: {action.rationale}",
                memories=analysis.memories,
                metrics=analysis.metrics,
            )

        return self.__build_plan_result(
            action=action,
            step_number=state.step_count,
            capture=capture,
            memories=analysis.memories,
            metrics=analysis.metrics,
        )

    def __select_action(
        self,
        state: AgentState,
        reasoner: Reasoner,
        analysis: AnalysisResult,
    ) -> Action:
        context = state.build_context()
        failures_raw = context.get("recent_failures", [])
        failures = failures_raw if isinstance(failures_raw, list) else []

        return reasoner.select_best_action(
            analysis.action,
            analysis.alternatives,
            failed_actions={str(failure) for failure in failures},
        )

    def __build_plan_result(
        self,
        action: Action,
        step_number: int,
        capture: ScreenCapture,
        *,
        is_recovery: bool = False,
        memories: int = 0,
        metrics: Optional[dict[str, float]] = None,
    ) -> PlanResult:
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
            memories=memories,
            metrics=metrics or {},
        )

    def __compute_simple_hash(self, capture: ScreenCapture) -> str:
        data = f"{capture.activity}:{len(capture.image)}".encode()
        return hashlib.md5(data, usedforsecurity=False).hexdigest()[:16]
