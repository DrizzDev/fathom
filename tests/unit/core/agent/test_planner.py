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
from fathom.schemas.subgoal import ExecutionContract, RequiredActionFamily, ScrollAxis, SubGoal


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


def _validation_completion_analysis() -> AnalysisResult:
    """
    Build an analysis result that uses VALIDATE as a terminal completion claim.
    """

    return AnalysisResult(
        outcome=AnalysisOutcome.ACT,
        reasoning="The target is visible; validate it one last time before completion.",
        screen_description="target card is visible",
        is_sub_goal_complete=True,
        action=Action(
            target="Jars & Containers visibility",
            confidence=1.0,
            action_type=ActionType.VALIDATE,
            rationale="Validate the visible target before completion.",
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
    async def test_surface_contract_is_forwarded_to_vision(self, memory_port_stub: Any) -> None:
        """
        Structured surface context must reach the vision layer unchanged so strict
        scroll plans keep targeting the requested area.
        """

        context_manager = ContextManager(memory=memory_port_stub, workflow_id="t3")
        state = AgentState(intent="find Millet Express")
        state.set_sub_goals(
            [
                SubGoal(
                    index=0,
                    description="Scroll horizontally below Fast Delivery until Millet Express is visible",
                    execution_contract=ExecutionContract(
                        required_action_family=RequiredActionFamily.SCROLL,
                        scroll_axis=ScrollAxis.HORIZONTAL,
                        surface="below Fast Delivery section",
                    ),
                )
            ]
        )

        vision = AsyncMock()
        vision.analyze.return_value = _analysis()

        planner = StepPlanner(vision_tool=vision)
        await planner.plan_step(
            state=state,
            capture=_capture(),
            reasoner=Reasoner(intent="find Millet Express"),
            screen_width=1206,
            screen_height=2622,
            context_manager=context_manager,
            strict_mode=True,
        )

        analyze_call = vision.analyze.await_args
        assert analyze_call is not None
        sub_goal_info = analyze_call.kwargs["sub_goal_info"]
        assert sub_goal_info["surface"] == "below Fast Delivery section"
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

    @pytest.mark.asyncio
    async def test_terminal_validation_candidate_is_tagged_on_step(
        self, memory_port_stub: Any
    ) -> None:
        """
        Planner must tag completion-claim validation steps so strict supervision can
        distinguish them from ordinary validation churn.
        """

        context_manager = ContextManager(memory=memory_port_stub, workflow_id="t4")
        state = AgentState(intent="find jars & containers")
        state.set_sub_goals(
            [
                SubGoal(
                    index=0,
                    description="Scroll down until you find Jars & containers on the screen",
                    execution_contract=ExecutionContract(
                        required_action_family=RequiredActionFamily.SCROLL,
                        scroll_axis=ScrollAxis.VERTICAL,
                    ),
                )
            ]
        )

        vision = AsyncMock()
        vision.analyze.return_value = _validation_completion_analysis()

        planner = StepPlanner(vision_tool=vision)
        plan = await planner.plan_step(
            state=state,
            capture=_capture(),
            reasoner=Reasoner(intent="find jars & containers"),
            screen_width=1206,
            screen_height=2622,
            context_manager=context_manager,
            strict_mode=True,
        )

        assert plan.step is not None
        assert plan.step.action.action_type is ActionType.VALIDATE
        assert plan.step.metadata["terminal_validation_candidate"] is True
        await context_manager.shutdown()
