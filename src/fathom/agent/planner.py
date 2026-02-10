from __future__ import annotations

import hashlib
import json
from logging import getLogger
import time
from typing import Any, Dict, Optional, Tuple

from fathom.agent.reasoner import Reasoner
from fathom.agent.state import AgentState
from fathom.schemas.actions import Action, Bounds
from fathom.schemas.results import AnalysisResult, PlanResult
from fathom.schemas.screens import ScreenCapture
from fathom.schemas.steps import Step
from fathom.tools.vision import VisionTool

logger = getLogger(name=__name__)
DEBUG_LOG_PATH = "/Users/mohnishbangaru/Fathom v1/fathom/.cursor/debug.log"


def _debug_log(hypothesis_id: str, location: str, message: str, data: Dict[str, Any]) -> None:
    payload = {
        "runId": "pre-fix",
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    try:
        with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as debug_file:
            debug_file.write(json.dumps(payload, ensure_ascii=True) + "\n")
    except Exception:
        logger.debug("Debug instrumentation write failed", exc_info=True)


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

        # region agent log
        _debug_log(
            hypothesis_id="H5",
            location="src/fathom/agent/planner.py:plan_step",
            message="Entering planner plan_step",
            data={
                "step_count": state.step_count,
                "is_complete": state.is_complete,
                "is_stuck": state.is_stuck,
                "can_continue": state.can_continue,
                "capture_activity": capture.activity,
            },
        )
        # endregion
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
            completion_signal = False
            completion_error = None
            try:
                completion_signal = await self.__vision.check_completion(
                    intent=state.intent, capture=capture
                )
            except Exception as exception:
                completion_error = str(exception)
            # region agent log
            _debug_log(
                hypothesis_id="H3",
                location="src/fathom/agent/planner.py:plan_step",
                message="Stuck gate reached before analysis path",
                data={
                    "is_stuck": True,
                    "can_recover": state.can_continue,
                    "capture_activity": capture.activity,
                    "check_completion_signal": completion_signal,
                    "check_completion_error": completion_error,
                },
            )
            # endregion

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

        action = self.__select_action(state=state, reasoner=reasoner, analysis=analysis)

        if completion.is_complete:
            state.mark_complete(reason=completion.evidence)
            
            # If there's a valid physical action, we should execute it before finishing
            step = None
            if action.action_type not in ("complete", "unknown", "wait"):
                 step = self.__build_step(
                     action=action, 
                     step_number=state.step_count, 
                     capture=capture
                 )

            return PlanResult(
                step=step,
                is_complete=True,
                metrics=analysis.metrics,
                metadata=analysis.metadata,
                reason=completion.evidence,
                memories=analysis.memories,
            )

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
            action=action,
            step_number=step_number,
            capture=capture,
            is_recovery=is_recovery
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
