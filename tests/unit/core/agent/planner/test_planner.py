from __future__ import annotations

import unittest
from typing import Tuple
from unittest.mock import AsyncMock, Mock

from tests.builders import ActionFixtures, AgentFixtures, ScreenFixtures, SubGoalFixtures

from fathom.constants import ActionType
from fathom.constants.retries import RetryBranch, RetryKind, RetryMetadataField
from fathom.constants.state import CompletionReason
from fathom.constants.tools import ToolName
from fathom.core.agent.planner import StepPlanner
from fathom.core.agent.state import AgentState
from fathom.schemas.actions import Action
from fathom.schemas.results import AnalysisResult, PlanResult
from fathom.schemas.supervision import BlockReason


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
            result.metadata.get(RetryMetadataField.BLOCK_REASON.value),
            BlockReason.REPEATED_CURRENT_SCREEN_ACTION.value,
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


class StepPlannerSilentRejectionBranchTest(unittest.IsolatedAsyncioTestCase):
    """
    Pins the should_avoid_action branch — the production loop on PFTXN run 77873149 returned should_retry=True from here 30 times with no LLM feedback and no budget consumption; both must now be present.
    """

    @staticmethod
    def __seed_failed_action(*, state: AgentState, descriptor_target: str) -> None:
        """
        Seed a same-target failure via the public ``record_repeated_action_failure`` helper so should_avoid_action returns True without reaching into private fields.
        """

        failed = Action(
            target=descriptor_target,
            confidence=1.0,
            rationale="seeded failure",
            action_type=ActionType.SWIPE_LEFT,
        )
        state.record_repeated_action_failure(action=failed)

    async def __plan(
        self,
        *,
        descriptor_target: str,
        rejection_history_return: object,
    ) -> Tuple[PlanResult, AgentState, Mock]:
        """
        Drive one ``plan_step`` round with a vision stub that surfaces the rejected action and a pre-seeded failure history.
        """

        action = ActionFixtures.make(
            target=descriptor_target,
            natural_language_target=descriptor_target,
            rationale="Scroll left to find Gourmet delights.",
            action_type=ActionType.SWIPE_LEFT,
        )
        analysis = AnalysisResult(
            action=action,
            reasoning="The carousel needs to scroll left.",
            screen_description="Home feed",
            metadata={"tool_args": {}},
        )
        vision = Mock()
        vision.analyze = AsyncMock(return_value=analysis)
        vision.build_rejection_history_from_analysis = Mock(return_value=rejection_history_return)

        state = AgentFixtures.state(intent="Find Gourmet delights")
        state.set_sub_goals(
            [SubGoalFixtures.make(description="Scroll left to find Gourmet delights")]
        )
        self.__seed_failed_action(state=state, descriptor_target=descriptor_target)

        reasoner = Mock()
        reasoner.select_best_action.return_value = action
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

        return result, state, vision

    async def test_should_avoid_action_returns_should_retry_with_no_step(self) -> None:
        """
        The silent branch must surface should_retry=True with step=None so the graph routes back to GROUND.
        """

        rejection_payload = ["seeded-turn"]
        result, _, _ = await self.__plan(
            descriptor_target="More on Swiggy widget",
            rejection_history_return=rejection_payload,
        )

        self.assertIsNone(result.step)
        self.assertTrue(result.should_retry)
        self.assertFalse(result.is_complete)

    async def test_should_avoid_action_writes_rejection_history(self) -> None:
        """
        The branch must seed conversation rejection_history so the next LLM turn sees its proposal was rejected; without this the LLM re-proposes the same action.
        """

        rejection_payload = ["seeded-turn"]
        _, state, vision = await self.__plan(
            descriptor_target="More on Swiggy widget",
            rejection_history_return=rejection_payload,
        )

        self.assertEqual(state.rejection_history, rejection_payload)
        vision.build_rejection_history_from_analysis.assert_called_once()
        kwargs = vision.build_rejection_history_from_analysis.call_args.kwargs
        self.assertIn("REJECTED", kwargs["rejection_reason"])
        # AgentFixtures.state defaults HITL off; non-interactive guidance must not advertise ask_user.
        self.assertNotIn("ask_user", kwargs["rejection_reason"])

    async def test_should_avoid_action_stamps_retry_metadata(self) -> None:
        """
        analyze.py routes the planner-retry budget off the metadata kind/branch keys; missing or stale values would route the wrong budget.
        """

        result, _, _ = await self.__plan(
            descriptor_target="More on Swiggy widget",
            rejection_history_return=["x"],
        )

        self.assertEqual(
            result.metadata.get(RetryMetadataField.KIND.value),
            RetryKind.SILENT_REJECTION.value,
        )
        self.assertEqual(
            result.metadata.get(RetryMetadataField.BRANCH.value),
            RetryBranch.SHOULD_AVOID_ACTION.value,
        )
        self.assertIn(
            "More on Swiggy widget",
            str(result.metadata.get(RetryMetadataField.BLOCKED_ACTION.value, "")),
        )

    async def test_should_avoid_action_does_not_advance_step_count(self) -> None:
        """
        The whole bug class hinges on ``step_count`` not advancing when the planner silently retries; this test pins that invariant.
        """

        before = AgentFixtures.state(intent="x").step_count
        _, state, _ = await self.__plan(
            descriptor_target="More on Swiggy widget",
            rejection_history_return=["x"],
        )
        self.assertEqual(state.step_count, before)


class StepPlannerTerminalReasonResolutionTest(unittest.IsolatedAsyncioTestCase):
    """
    Pins that the planner surfaces the state's authoritative completion_reason when the state has already been marked complete by a non-success path (FAILED/CANCELLED/MAX_STEPS).
    """

    async def __plan(self, *, state: AgentState) -> PlanResult:
        """
        Drive a single ``plan_step`` against the supplied state with a benign vision stub.
        """

        planner = StepPlanner(vision_tool=Mock())
        return await planner.plan_step(
            state=state,
            reasoner=Mock(),
            capture=ScreenFixtures.capture(activity="app"),
            context_manager=AgentFixtures.context_manager(),
            screen_width=100,
            screen_height=200,
            prompt_if_stuck=False,
        )

    async def test_state_marked_cancelled_preserves_cancelled_reason(self) -> None:
        """
        A state already marked complete with CANCELLED must not be reported back to the router as SUCCESS.
        """

        state = AgentFixtures.state(intent="x")
        state.mark_complete(reason=CompletionReason.CANCELLED.value)

        result = await self.__plan(state=state)

        self.assertTrue(result.is_complete)
        self.assertEqual(result.reason, CompletionReason.CANCELLED.value)

    async def test_state_marked_failed_preserves_failed_reason(self) -> None:
        """
        A state already marked complete with FAILED must not be masked as SUCCESS by the terminal-reason resolver.
        """

        state = AgentFixtures.state(intent="x")
        state.mark_complete(reason=CompletionReason.FAILED.value)

        result = await self.__plan(state=state)

        self.assertTrue(result.is_complete)
        self.assertEqual(result.reason, CompletionReason.FAILED.value)

    async def test_pre_exhausted_retry_state_marks_agent_state_complete(self) -> None:
        """
        A checkpoint restored with an already-exhausted planner-retry budget short-circuits at ``can_continue=False``; the planner must mark the AgentState complete so the next persist/restore does not see an incomplete workflow that would try to continue.
        """

        state = AgentFixtures.state(intent="x")
        for _ in range(state.retries.planner.cap):
            state.tick_planner_retry(
                kind=RetryKind.SILENT_REJECTION,
                branch=RetryBranch.SHOULD_AVOID_ACTION,
                action="Swipe left",
            )
        self.assertTrue(state.retries.planner.exhausted)
        self.assertFalse(state.is_complete)

        result = await self.__plan(state=state)

        self.assertTrue(result.is_complete)
        self.assertEqual(result.reason, CompletionReason.RETRY_BUDGET_EXHAUSTED.value)
        self.assertTrue(state.is_complete)
        self.assertEqual(state.completion_reason, CompletionReason.RETRY_BUDGET_EXHAUSTED.value)

    async def test_state_marked_complete_without_reason_defaults_to_success(self) -> None:
        """
        Backward-compat: a state marked complete with no explicit reason defaults to SUCCESS.
        """

        state = AgentFixtures.state(intent="x")
        state.mark_complete(reason="")

        result = await self.__plan(state=state)

        self.assertTrue(result.is_complete)
        self.assertEqual(result.reason, CompletionReason.SUCCESS.value)
