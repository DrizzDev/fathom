"""
Pins that :meth:`StepPlanner.plan_step` consumes user guidance once
and clears it from the channel — same lifecycle as verifier feedback.

Healing.txt (workflow ``a06c91a3``) surfaced a 6-step cascade where a
single HITL nudge ("Tap on the cross icon") survived in
``ContextManager.user_guidance`` across every later ANALYZE call. The
planner kept passing it through to the LLM long after the original
overlay was dismissed, so the agent kept emitting "Tap on the cross
icon" against unrelated screens. The fix: planner clears
``user_guidance`` after a successful analyze the same way it already
clears ``verifier_feedback`` and ``rejection_history``.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from fathom.constants import ActionType
from fathom.core.agent.planner import StepPlanner
from fathom.core.agent.reasoner import Reasoner
from fathom.core.agent.state import AgentState
from fathom.core.context.manager import ContextManager
from fathom.schemas.actions import Action
from fathom.schemas.results import AnalysisOutcome, AnalysisResult
from fathom.schemas.screens import ScreenCapture
from fathom.schemas.subgoal import SubGoal


def _capture() -> ScreenCapture:
    """
    Build a minimal :class:`ScreenCapture` for planner unit tests.
    """

    return ScreenCapture(
        width=1206,
        height=2622,
        timestamp=0,
        image=b"png-bytes",
        activity="bundl.swiggy.production",
    )


def _analysis(*, outcome: AnalysisOutcome = AnalysisOutcome.ACT) -> AnalysisResult:
    """
    Build a minimal :class:`AnalysisResult` with a benign tap action so
    the planner's downstream blocking checks (``should_avoid_action``,
    ``is_action_repeating_on_screen``) do not short-circuit before the
    use-once clears fire.
    """

    return AnalysisResult(
        outcome=outcome,
        reasoning="tap the visible CTA",
        screen_description="search results page",
        action=Action(
            target="20",
            label_id="20",
            confidence=0.9,
            action_type=ActionType.TAP,
            rationale="tap the visible CTA",
        ),
    )


class TestPlannerUseOnceGuidance:
    """
    Behavioural pins for planner-side use-once consumption of user
    guidance.
    """

    @pytest.mark.asyncio
    async def test_guidance_cleared_after_successful_analysis(self, memory_port_stub: Any) -> None:
        """
        A single ``plan_step`` cycle consumes user guidance and leaves
        the channel empty — preventing the healing.txt stickiness.
        """

        context_manager = ContextManager(memory=memory_port_stub, workflow_id="t1")
        await context_manager.inject_user_guidance(guidance="Tap on the cross icon")
        assert len(context_manager.get_user_guidance()) == 1

        state = AgentState(intent="order dosa")
        state.set_sub_goals([SubGoal(index=0, description="Tap on the cross icon")])

        vision = AsyncMock()
        vision.analyze.return_value = _analysis()

        planner = StepPlanner(vision_tool=vision)
        await planner.plan_step(
            state=state,
            capture=_capture(),
            reasoner=Reasoner(intent="order dosa"),
            screen_width=1206,
            screen_height=2622,
            context_manager=context_manager,
        )

        assert context_manager.get_user_guidance() == []
        await context_manager.shutdown()

    @pytest.mark.asyncio
    async def test_clear_is_noop_when_no_guidance_present(self, memory_port_stub: Any) -> None:
        """
        Calling ``plan_step`` without any injected guidance must not
        error — the clear path is idempotent.
        """

        context_manager = ContextManager(memory=memory_port_stub, workflow_id="t2")
        state = AgentState(intent="order dosa")
        state.set_sub_goals([SubGoal(index=0, description="Tap on Skip button")])

        vision = AsyncMock()
        vision.analyze.return_value = _analysis()

        planner = StepPlanner(vision_tool=vision)
        await planner.plan_step(
            state=state,
            capture=_capture(),
            reasoner=Reasoner(intent="order dosa"),
            screen_width=1206,
            screen_height=2622,
            context_manager=context_manager,
        )

        assert context_manager.get_user_guidance() == []
        await context_manager.shutdown()
