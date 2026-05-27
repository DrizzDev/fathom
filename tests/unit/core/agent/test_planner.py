from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from fathom.constants import ActionType
from fathom.constants.state import CompletionReason
from fathom.core.agent.planner import StepPlanner
from fathom.core.agent.state import AgentState
from fathom.schemas.actions import Action
from fathom.schemas.capabilities import HITLCapability, RuntimeCapabilities
from fathom.schemas.results import AnalysisResult
from fathom.schemas.screens import ScreenCapture, ScreenState
from fathom.schemas.subgoal import SubGoal


class StepPlannerStuckFlowTest(unittest.IsolatedAsyncioTestCase):
    """
    Covers stuck-state planning across autonomous and HITL mode.
    """

    @staticmethod
    def __screen() -> ScreenState:
        """
        Return a stable screen used to create loop-detector evidence.
        """

        return ScreenState(
            activity="app",
            timestamp=0,
            activity_hash="a" * 16,
            visual_hash="b" * 16,
        )

    @staticmethod
    def __capture() -> ScreenCapture:
        """
        Return a minimal screen capture for step construction.
        """

        return ScreenCapture(
            width=100,
            height=200,
            activity="app",
            image=b"png",
            timestamp=1,
        )

    @staticmethod
    def __context() -> SimpleNamespace:
        """
        Return the context-manager surface used before ASK_USER.
        """

        return SimpleNamespace(
            get_user_guidance=Mock(return_value=[]),
            consume_user_guidance=Mock(),
            clear_user_guidance=Mock(),
            clear_verifier_feedback=Mock(),
        )

    def __stuck_state_with_exhausted_autonomous_budget(self) -> AgentState:
        """
        Build a state whose native recovery budget is exhausted.
        """

        state = AgentState(
            intent="finish onboarding",
            capabilities=RuntimeCapabilities(hitl=HITLCapability(enabled=True)),
        )
        detector = state.runtime.screen.detector
        for _ in range(detector.threshold):
            detector.record(
                screen=self.__screen(),
                action_type="tap",
                action_description="Tap Continue",
            )

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
            capture=self.__capture(),
            context_manager=self.__context(),
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

        action = Action(
            action_type=ActionType.TAP,
            target="HSR Layout option",
            natural_language_target="HSR Layout option",
            rationale="Tap the previous location option.",
            confidence=0.9,
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

        state = AgentState(
            intent="Tap on search bar",
            capabilities=RuntimeCapabilities(hitl=HITLCapability(enabled=False)),
        )
        state.set_sub_goals([SubGoal(index=0, description="Tap on search bar")])
        planner = StepPlanner(vision_tool=vision)
        reasoner = Mock()
        reasoner.select_best_action.return_value = action

        result = await planner.plan_step(
            state=state,
            reasoner=reasoner,
            capture=self.__capture(),
            context_manager=self.__context(),
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

        action = Action(
            action_type=ActionType.TAP,
            target="Swiggy app icon",
            natural_language_target="Swiggy app icon",
            rationale="Open the app.",
            confidence=1.0,
        )
        analysis = AnalysisResult(
            action=action,
            reasoning="The app icon is visible.",
            screen_description="Launcher",
            metadata={
                "previous_screen_actions": [
                    {"success": True, "action": "tap", "target": "Swiggy app icon"}
                ],
                "tool_args": {},
            },
        )
        vision = Mock()
        vision.analyze = AsyncMock(return_value=analysis)

        state = AgentState(
            intent="Open Swiggy app",
            capabilities=RuntimeCapabilities(hitl=HITLCapability(enabled=False)),
        )
        state.set_sub_goals([SubGoal(index=0, description="Open Swiggy app")])
        planner = StepPlanner(vision_tool=vision)
        reasoner = Mock()
        reasoner.select_best_action.return_value = action

        result = await planner.plan_step(
            state=state,
            reasoner=reasoner,
            capture=self.__capture(),
            context_manager=self.__context(),
            screen_width=100,
            screen_height=200,
            prompt_if_stuck=False,
        )

        self.assertIsNotNone(result.step)
        self.assertFalse(result.should_retry)


class StepPlannerAutonomousAskUserSubstitutionTest(unittest.IsolatedAsyncioTestCase):
    """Pins the autonomous-runtime substitution: ASK_USER -> recovery ladder."""

    @staticmethod
    def __capture() -> ScreenCapture:
        """Return a minimal screen capture."""

        return ScreenCapture(
            width=100,
            height=200,
            activity="app",
            image=b"png",
            timestamp=1,
        )

    @staticmethod
    def __context() -> SimpleNamespace:
        """Return the context-manager surface used by the planner."""

        return SimpleNamespace(
            get_user_guidance=Mock(return_value=[]),
            inject_user_guidance=AsyncMock(),
            consume_user_guidance=Mock(),
            clear_user_guidance=Mock(),
            clear_verifier_feedback=Mock(),
        )

    @staticmethod
    def __ask_user_analysis() -> AnalysisResult:
        """Return an analysis whose primary action is ASK_USER."""

        action = Action(
            action_type=ActionType.ASK_USER,
            target="User",
            rationale="missing credentials",
            confidence=0.9,
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

        state = AgentState(
            intent="login",
            capabilities=RuntimeCapabilities(hitl=HITLCapability(enabled=False)),
        )
        state.set_sub_goals([SubGoal(description="login", index=0)])
        reasoner = Mock()
        reasoner.select_best_action.return_value = analysis.action

        planner = StepPlanner(vision_tool=vision)
        result = await planner.plan_step(
            state=state,
            reasoner=reasoner,
            capture=self.__capture(),
            context_manager=self.__context(),
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

        state = AgentState(
            intent="login",
            capabilities=RuntimeCapabilities(hitl=HITLCapability(enabled=False)),
        )
        state.set_sub_goals([SubGoal(description="login", index=0)])
        while state.runtime.screen.detector.can_recover():
            state.runtime.screen.detector.record_recovery_attempt()

        reasoner = Mock()
        reasoner.select_best_action.return_value = analysis.action

        planner = StepPlanner(vision_tool=vision)
        result = await planner.plan_step(
            state=state,
            reasoner=reasoner,
            capture=self.__capture(),
            context_manager=self.__context(),
            screen_width=100,
            screen_height=200,
            prompt_if_stuck=False,
        )

        self.assertIsNone(result.step)
        self.assertTrue(result.is_complete)
        self.assertEqual(result.reason, CompletionReason.INTERVENTION_REQUIRED.value)
