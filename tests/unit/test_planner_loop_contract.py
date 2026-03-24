from unittest.mock import AsyncMock, MagicMock

import pytest

from fathom.agent.planner import StepPlanner
from fathom.agent.reasoner import Reasoner
from fathom.agent.state import AgentState
from fathom.constants import ActionType
from fathom.schemas.actions import Action
from fathom.schemas.results import AnalysisResult
from fathom.schemas.screens import ScreenCapture, ScreenState


def _capture() -> ScreenCapture:
    return ScreenCapture(
        width=1080,
        height=2400,
        activity="com.example/.MainActivity",
        image=b"\x89PNG\r\n\x1a\n" + b"\x00" * 100,
        timestamp=1000000,
        state=ScreenState(
            activity="com.example/.MainActivity",
            timestamp=1000000,
            activity_hash="abcd1234",
            structural_hash="struct1234",
            visual_hash="11112222",
        ),
    )


def _analysis() -> AnalysisResult:
    return AnalysisResult(
        action=Action(
            action_type=ActionType.TAP,
            rationale="Tap the search bar",
            target="search bar",
            natural_language_target="search bar",
            confidence=0.9,
        ),
        alternatives=[],
        reasoning="Next best action",
        screen_description="Home screen with search",
        is_goal_complete=False,
    )


@pytest.mark.asyncio
async def test_plan_step_passes_delta_context_to_vision():
    vision = MagicMock()
    vision.analyze = AsyncMock(return_value=_analysis())
    planner = StepPlanner(vision_tool=vision)
    state = AgentState(intent="search for milk", max_steps=5)
    reasoner = Reasoner(intent="search for milk")

    await planner.plan_step(
        state=state,
        reasoner=reasoner,
        capture=_capture(),
    )

    kwargs = vision.analyze.await_args.kwargs
    assert "delta_context" in kwargs
    assert isinstance(kwargs["delta_context"], dict)
