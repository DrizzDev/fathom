from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, Mock

from tests.builders import ActionFixtures, AgentFixtures, ScreenFixtures, SubGoalFixtures

from fathom.constants import ActionType
from fathom.constants.state import CompletionReason
from fathom.constants.tools import ToolName
from fathom.core.agent.planner import StepPlanner
from fathom.core.agent.state import AgentState
from fathom.schemas.results import AnalysisResult


class StepPlannerStuckFlowTest(unittest.IsolatedAsyncioTestCase):
    """
    Covers stuck-state planning across autonomous and HITL mode.
    """

    def __stuck_state_with_exhausted_autonomous_budget(self) -> AgentState:
        """
        Build a state whose native recovery budget is exhausted.
        """

        state = AgentFixtures.stuck_state(intent="finish onboarding", hitl_enabled=True)
        detector = state.runtime.screen.detector
        while detector.can_recover():
            detector.record_recovery_attempt()
        return state

    async def test_interactive_stuck_flow_asks_user_after_autonomous_budget_exhausted(self) -> None:
        """
        HITL mode must reach ASK_USER even after autonomous recovery is exhausted.
        """

        planner = StepPlanner(vision_tool=Mock())
        result = await planner.plan_step(
            state=self.__stuck_state_with_exhausted_autonomous_budget(),
            reasoner=Mock(),
            capture=ScreenFixtures.capture(activity="app"),
            context_manager=AgentFixtures.context_manager(),
            screen_width=100,
            screen_height=200,
            prompt_if_stuck=True,
        )

        self.assertFalse(result.is_complete)
        self.assertIsNotNone(result.step)
        assert result.step is not None
        self.assertEqual(result.step.action.action_type, ActionType.ASK_USER)

    async def test_blocks_successful_current_screen_action_repeat(self) -> None:
        """
        Planner rejects an action that repeats a successful action from current-screen memory.
        """

        action = ActionFixtures.make(
            target="HSR Layout option",
            natural_language_target="HSR Layout option",
            rationale="Tap the previous location option.",
        )
        analysis = AnalysisResult(
            action=action,
            reasoning="The option is visible.",
            screen_description="Select Your Location",
            metadata={
                "current_workflow_screen_actions": [
                    {"success": True, "action": "tap", "target": "HSR Layout option"}
                ],
                "tool_args": {},
            },
        )
        vision = Mock()
        vision.analyze = AsyncMock(return_value=analysis)
        vision.build_rejection_history_from_analysis = Mock(return_value=[])

        state = AgentFixtures.state(intent="Tap on search bar")
        state.set_sub_goals([SubGoalFixtures.make(description="Tap on search bar")])
        planner = StepPlanner(vision_tool=vision)
        reasoner = Mock()
        reasoner.select_best_action.return_value = action

        result = await planner.plan_step(
            state=state,
            reasoner=reasoner,
            capture=ScreenFixtures.capture(activity="app"),
            context_manager=AgentFixtures.context_manager(),
            screen_width=100,
            screen_height=200,
            prompt_if_stuck=False,
        )

        self.assertIsNone(result.step)
        self.assertTrue(result.should_retry)
        self.assertEqual(result.reason, CompletionReason.ACTION_BLOCKED.value)
        self.assertEqual(
            result.metadata.get("block_reason"),
            "repeated_current_screen_action",
        )

    async def test_does_not_block_persistent_screen_memory(self) -> None:
        """
        Persistent screen memories must not be treated as current-workflow repeats.
        """

        action = ActionFixtures.make(
            target="App icon",
            natural_language_target="App icon",
            rationale="Open the app.",
            confidence=1.0,
        )
        analysis = AnalysisResult(
            action=action,
            reasoning="The app icon is visible.",
            screen_description="Launcher",
            metadata={
                "previous_screen_actions": [
                    {"success": True, "action": "tap", "target": "App icon"}
                ],
                "tool_args": {},
            },
        )
        vision = Mock()
        vision.analyze = AsyncMock(return_value=analysis)

        state = AgentFixtures.state(intent="Open app")
        state.set_sub_goals([SubGoalFixtures.make(description="Open app")])
        planner = StepPlanner(vision_tool=vision)
        reasoner = Mock()
        reasoner.select_best_action.return_value = action

        result = await planner.plan_step(
            state=state,
            reasoner=reasoner,
            capture=ScreenFixtures.capture(activity="app"),
            context_manager=AgentFixtures.context_manager(),
            screen_width=100,
            screen_height=200,
            prompt_if_stuck=False,
        )

        self.assertIsNotNone(result.step)
        self.assertFalse(result.should_retry)


class StepPlannerToolScopeTest(unittest.IsolatedAsyncioTestCase):
    """
    Pins planner-to-tool-scope mapping for decomposed and non-decomposed turns.
    """

    async def test_no_sub_goal_turn_exposes_verification_tools(self) -> None:
        """
        Non-decomposed planner turns retain whole-goal verification capability.
        """

        action = ActionFixtures.make(rationale="Tap visible continue button.")
        analysis = AnalysisResult(
            action=action,
            reasoning="Continue is visible.",
            screen_description="Onboarding",
            metadata={"tool_args": {}},
        )
        vision = Mock()
        vision.analyze = AsyncMock(return_value=analysis)

        reasoner = Mock()
        reasoner.select_best_action.return_value = action
        reasoner.analyze_completion.return_value = Mock(is_complete=False)

        await StepPlanner(vision_tool=vision).plan_step(
            state=AgentFixtures.state(intent="verify offerwall is open"),
            reasoner=reasoner,
            capture=ScreenFixtures.capture(activity="app"),
            context_manager=AgentFixtures.context_manager(),
            screen_width=100,
            screen_height=200,
        )

        tools = vision.analyze.call_args.kwargs["tools"]
        self.assertIn(ToolName.VERIFY_GOAL, tools.names)
        self.assertIn(ToolName.VALIDATE_STATE, tools.names)

    async def test_action_sub_goal_turn_hides_verification_tools(self) -> None:
        """
        ACTION sub-goals keep verification tools hidden even when the intent says verify.
        """

        action = ActionFixtures.make(rationale="Tap Play.")
        analysis = AnalysisResult(
            action=action,
            reasoning="Play is visible.",
            screen_description="Game home",
            metadata={"tool_args": {}},
        )
        vision = Mock()
        vision.analyze = AsyncMock(return_value=analysis)

        state = AgentFixtures.state(intent="open the game and verify offerwall")
        state.set_sub_goals([SubGoalFixtures.make(description="Press Play")])

        reasoner = Mock()
        reasoner.select_best_action.return_value = action

        await StepPlanner(vision_tool=vision).plan_step(
            state=state,
            reasoner=reasoner,
            capture=ScreenFixtures.capture(activity="app"),
            context_manager=AgentFixtures.context_manager(),
            screen_width=100,
            screen_height=200,
        )

        tools = vision.analyze.call_args.kwargs["tools"]
        self.assertNotIn(ToolName.VERIFY_GOAL, tools.names)
        self.assertNotIn(ToolName.VALIDATE_STATE, tools.names)


class StepPlannerAutonomousAskUserSubstitutionTest(unittest.IsolatedAsyncioTestCase):
    """Pins the autonomous-runtime substitution: ASK_USER -> recovery ladder."""

    @staticmethod
    def __ask_user_analysis() -> AnalysisResult:
        """Return an analysis whose primary action is ASK_USER."""

        action = ActionFixtures.make(
            target="User",
            action_type=ActionType.ASK_USER,
            rationale="missing credentials",
            text="What is your OTP?",
        )
        return AnalysisResult(
            action=action,
            reasoning="Need credentials",
            screen_description="login",
            metadata={"tool_args": {}},
        )

    async def test_substitutes_ask_user_with_recovery_action(self) -> None:
        """ASK_USER on autonomous runtime substitutes with the next ladder rung."""

        analysis = self.__ask_user_analysis()
        vision = Mock()
        vision.analyze = AsyncMock(return_value=analysis)
        vision.build_rejection_history_from_analysis = Mock(return_value=[])

        state = AgentFixtures.state(intent="login")
        state.set_sub_goals([SubGoalFixtures.make(description="login")])
        reasoner = Mock()
        reasoner.select_best_action.return_value = analysis.action

        planner = StepPlanner(vision_tool=vision)
        result = await planner.plan_step(
            state=state,
            reasoner=reasoner,
            capture=ScreenFixtures.capture(activity="app"),
            context_manager=AgentFixtures.context_manager(),
            screen_width=100,
            screen_height=200,
            prompt_if_stuck=False,
        )

        self.assertIsNotNone(result.step)
        assert result.step is not None
        self.assertNotEqual(result.step.action.action_type, ActionType.ASK_USER)
        self.assertIn(
            result.step.action.action_type,
            {ActionType.BACK, ActionType.SCROLL, ActionType.HOME},
        )
        self.assertFalse(result.is_complete)

    async def test_terminates_when_ladder_exhausted(self) -> None:
        """When the recovery ladder is spent, ASK_USER becomes terminal INTERVENTION_REQUIRED."""

        analysis = self.__ask_user_analysis()
        vision = Mock()
        vision.analyze = AsyncMock(return_value=analysis)
        vision.build_rejection_history_from_analysis = Mock(return_value=[])

        state = AgentFixtures.state(intent="login")
        state.set_sub_goals([SubGoalFixtures.make(description="login")])
        while state.runtime.screen.detector.can_recover():
            state.runtime.screen.detector.record_recovery_attempt()

        reasoner = Mock()
        reasoner.select_best_action.return_value = analysis.action

        planner = StepPlanner(vision_tool=vision)
        result = await planner.plan_step(
            state=state,
            reasoner=reasoner,
            capture=ScreenFixtures.capture(activity="app"),
            context_manager=AgentFixtures.context_manager(),
            screen_width=100,
            screen_height=200,
            prompt_if_stuck=False,
        )

        self.assertIsNone(result.step)
        self.assertTrue(result.is_complete)
        self.assertEqual(result.reason, CompletionReason.INTERVENTION_REQUIRED.value)
