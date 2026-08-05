from __future__ import annotations

from typing import Any, Dict, List, Optional, cast

from fathom.core.agent.state import AgentState
from fathom.core.context.manager import ContextManager
from fathom.core.services.vision import SubGoalContext, VisionService
from fathom.schemas.observation import ScreenObservation
from fathom.schemas.results import AnalysisResult
from fathom.schemas.screens import ScreenCapture
from fathom.schemas.subgoal import GoalState
from fathom.schemas.success import ObservedSuccess
from fathom.schemas.tools import AllowedTools


class VisionTurn:
    """
    Performs the single provider-neutral VLM request for the turn against the resolved tool scope.
    """

    def __init__(self, *, vision: VisionService) -> None:
        """
        Bind the vision service the turn requests against.
        """

        self.__vision = vision

    async def analyze(
        self,
        *,
        state: AgentState,
        capture: ScreenCapture,
        context_manager: ContextManager,
        tools: AllowedTools,
        screen_width: int,
        screen_height: int,
        use_xml: bool,
        elements: Optional[Dict[str, Any]],
        screen_observation: Optional[ScreenObservation],
        tracking_note: Optional[str],
    ) -> AnalysisResult:
        """
        Request one vision analysis for the active goal against the current screen with the resolved tools.
        """

        return await self.__vision.analyze(
            use_xml=use_xml,
            capture=capture,
            elements=elements,
            tools=tools,
            intent=state.intent,
            is_stuck=state.is_stuck,
            screen_width=screen_width,
            screen_height=screen_height,
            sub_goal_info=self.__sub_goal_context(state=state),
            context_manager=context_manager,
            last_action=state.last_action_type,
            tracking_note=tracking_note,
            screen_observation=screen_observation,
            loop_observation=state.build_loop_observation(),
            prior_rejection_history=state.rejection_history,
            visual_hash=capture.identity,
            failures=cast("List[str]", state.build_context().get("relevant_failures", [])),
        )

    @staticmethod
    def __sub_goal_context(*, state: AgentState) -> Optional[SubGoalContext]:
        """
        Build the minimal sub-goal context for vision, avoiding passing AgentState into the service.
        """

        current_sub_goal: Optional[GoalState] = state.get_current_sub_goal()
        if not (current_sub_goal and state.has_sub_goals()):
            return None

        current_idx, total = state.get_sub_goal_progress()
        success = current_sub_goal.success
        context = SubGoalContext(
            total=total,
            index=current_idx,
            description=current_sub_goal.objective,
            durable=not isinstance(success, ObservedSuccess),
        )
        if isinstance(success, ObservedSuccess):
            context["assertion"] = success.observation.assertion
        return context
