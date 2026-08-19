from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from tests.builders import SubGoalFixtures

from fathom.constants import ActionType
from fathom.constants.state import CompletionReason
from fathom.core.agent.planner import StepPlanner
from fathom.core.agent.state import AgentState
from fathom.schemas.actions import Action
from fathom.schemas.capabilities import HITLCapability, RuntimeCapabilities
from fathom.schemas.effect import ActionEffectStatus
from fathom.schemas.escalation import EscalationPolicy
from fathom.schemas.loop import LoopReason
from fathom.schemas.results import AnalysisResult
from fathom.schemas.screens import ScreenCapture, ScreenState


class EscalationProductionScenarioIntegrationTest(unittest.IsolatedAsyncioTestCase):
    """
    End-to-end replay of the production false-positive trail.
    """

    @staticmethod
    def __screen(*, visual_hash: str = "9ec0c0e4") -> ScreenState:
        return ScreenState(
            activity="com.foodapp/.SearchActivity",
            timestamp=0,
            activity_hash="search",
            visual_hash=visual_hash + "f" * (16 - len(visual_hash)),
            xml_hash="x",
            interaction_hash="i",
        )

    @staticmethod
    def __capture() -> ScreenCapture:
        return ScreenCapture(
            width=1080,
            height=2400,
            activity="com.foodapp/.SearchActivity",
            image=b"png",
            timestamp=1,
        )

    @staticmethod
    def __context() -> SimpleNamespace:
        return SimpleNamespace(
            get_user_guidance=Mock(return_value=[]),
            inject_user_guidance=AsyncMock(),
            consume_user_guidance=Mock(),
            clear_user_guidance=Mock(),
            clear_verifier_feedback=Mock(),
        )

    @staticmethod
    def __vision_with_navigation() -> Mock:
        """
        Stand-in vision service returning a benign TAP analysis on fallthrough.

        After deferral, the gate falls through to vision.analyze. We let the
        model return a tap so the planning result is well-formed; the goal of
        the integration test is to pin the escalation contract, not the
        post-deferral action choice.
        """

        action = Action(
            action_type=ActionType.TAP,
            target="Restaurant card",
            rationale="Tap on a result to verify the listing.",
            confidence=0.85,
        )
        analysis = AnalysisResult(
            action=action,
            reasoning="A restaurant card is visible; tap to verify.",
            screen_description="Search results page with restaurant cards.",
            metadata={"tool_args": {}},
        )
        vision = Mock()
        vision.analyze = AsyncMock(return_value=analysis)
        vision.build_rejection_history_from_analysis = Mock(return_value=[])
        return vision

    def __seed_validate_only_loop(self, *, state: AgentState) -> None:
        """
        Drive the detector into ``stuck=True`` purely via validate-only turns.

        Mirrors the production trail: two consecutive ``validate`` actions on
        the same Search Results Page hash, both classified NO_PROGRESS.
        """

        detector = state.runtime.screen.detector
        screen = self.__screen()
        for _ in range(detector.threshold):
            detector.record(
                screen=screen,
                action_type="validate",
                action_description="validate srp page",
                effect_status=ActionEffectStatus.NO_PROGRESS,
            )

    async def test_first_stuck_signal_defers_not_escalates(self) -> None:
        """
        Step 12 with no prior deferrals: gate defers, no ASK_USER returned.
        """

        state = AgentState(
            intent="search for restaurants near me",
            capabilities=RuntimeCapabilities(hitl=HITLCapability(enabled=True)),
        )
        state.set_sub_goals(
            [SubGoalFixtures.make(description="Validate srp page is loaded", index=0)]
        )
        self.__seed_validate_only_loop(state=state)

        # Confirm precondition: detector classifies the window as stuck.
        evidence = state.loop_evidence()
        self.assertTrue(evidence.stuck)
        self.assertIs(evidence.reason, LoopReason.INERT_REPETITION)
        # Contributing tail is validate-only NO_PROGRESS — the protected pattern.
        self.assertGreaterEqual(len(evidence.since_progress), 2)

        vision = self.__vision_with_navigation()
        reasoner = Mock()
        reasoner.select_best_action.return_value = vision.analyze.return_value.action
        planner = StepPlanner(vision_tool=vision)
        context = self.__context()

        result = (
            await planner.plan_step(
                state=state,
                reasoner=reasoner,
                capture=self.__capture(),
                context_manager=context,
                screen_width=1080,
                screen_height=2400,
                prompt_if_stuck=True,
            )
        ).plan

        # ASK_USER was NOT produced; the planner fell through to analysis.
        self.assertEqual(state.deferral_count, 1)
        context.inject_user_guidance.assert_awaited()
        vision.analyze.assert_awaited()
        # And the result is whatever the fallthrough produced — not an HITL.
        if result.step is not None:
            self.assertIsNot(result.step.action.action_type, ActionType.ASK_USER)

    async def test_repeated_deferrals_eventually_escape_to_ask_user(self) -> None:
        """
        After the deferral limit is exceeded, the gate escalates to ASK_USER.

        Default policy: deferral_limit=2. The third stuck-signal evaluation
        finds deferrals=3 > 2 and returns ASK_USER (DEFERRAL_LIMIT reason).
        """

        state = AgentState(
            intent="x", capabilities=RuntimeCapabilities(hitl=HITLCapability(enabled=True))
        )
        state.set_sub_goals([SubGoalFixtures.make(description="Validate srp page", index=0)])
        self.__seed_validate_only_loop(state=state)
        # Simulate two earlier deferrals on this sub-goal.
        state.record_deferral()
        state.record_deferral()
        state.record_deferral()  # Now at 3 (above default limit of 2).

        planner = StepPlanner(vision_tool=Mock())
        result = (
            await planner.plan_step(
                state=state,
                reasoner=Mock(),
                capture=self.__capture(),
                context_manager=self.__context(),
                screen_width=1080,
                screen_height=2400,
                prompt_if_stuck=True,
            )
        ).plan

        self.assertIsNotNone(result.step)
        assert result.step is not None
        self.assertIs(result.step.action.action_type, ActionType.ASK_USER)
        self.assertEqual(result.reason, CompletionReason.INTERVENTION_REQUIRED.value)
        # On allow, deferrals are cleared so the next sub-goal starts fresh.
        self.assertEqual(state.deferral_count, 0)

    async def test_passive_tolerance_exceeded_escalates(self) -> None:
        """
        Past tolerance=3 the gate escalates with PASSIVE_LIMIT reason.
        """

        state = AgentState(
            intent="x", capabilities=RuntimeCapabilities(hitl=HITLCapability(enabled=True))
        )
        state.set_sub_goals([SubGoalFixtures.make(description="Validate srp page", index=0)])
        detector = state.runtime.screen.detector
        for _ in range(4):
            detector.record(
                screen=self.__screen(),
                action_type="validate",
                action_description="validate",
                effect_status=ActionEffectStatus.NO_PROGRESS,
            )

        planner = StepPlanner(vision_tool=Mock())
        result = (
            await planner.plan_step(
                state=state,
                reasoner=Mock(),
                capture=self.__capture(),
                context_manager=self.__context(),
                screen_width=1080,
                screen_height=2400,
                prompt_if_stuck=True,
            )
        ).plan

        self.assertIsNotNone(result.step)
        assert result.step is not None
        self.assertIs(result.step.action.action_type, ActionType.ASK_USER)

    async def test_subgoal_budget_exhaustion_escalates(self) -> None:
        """
        Sub-goal budget exhaustion reaches the gate independently of is_stuck.
        """

        state = AgentState(
            intent="x", capabilities=RuntimeCapabilities(hitl=HITLCapability(enabled=True))
        )
        # max_steps=1 means a single recorded action exhausts the budget.
        state.set_sub_goals([SubGoalFixtures.make(description="active", index=0)])
        current = state.get_current_sub_goal()
        assert current is not None
        current.progress.limit = 1
        state.record_sub_goal_action()

        self.assertFalse(state.is_stuck)
        self.assertTrue(state.current_sub_goal_over_budget)

        planner = StepPlanner(vision_tool=Mock())
        result = (
            await planner.plan_step(
                state=state,
                reasoner=Mock(),
                capture=self.__capture(),
                context_manager=self.__context(),
                screen_width=1080,
                screen_height=2400,
                prompt_if_stuck=True,
            )
        ).plan

        self.assertIsNotNone(result.step)
        assert result.step is not None
        self.assertIs(result.step.action.action_type, ActionType.ASK_USER)

    async def test_deferral_resets_on_observable_progress(self) -> None:
        """
        A subsequent successful navigation with screen change clears the count.
        """

        from fathom.schemas.steps import Step, StepResult

        state = AgentState(
            intent="x", capabilities=RuntimeCapabilities(hitl=HITLCapability(enabled=True))
        )
        state.set_sub_goals([SubGoalFixtures.make(description="v", index=0)])
        state.record_deferral()
        state.record_deferral()
        self.assertEqual(state.deferral_count, 2)

        action = Action(
            action_type=ActionType.TAP,
            target="Restaurant card",
            rationale="navigate",
            confidence=0.9,
        )
        step = Step(action=action, step_number=0, screen_hash="pre")
        progress_result = StepResult(
            step=step,
            success=True,
            duration=10,
            screen_changed=True,
            pre_hash="pre",
            post_hash="post",
        )
        state.record_step(result=progress_result)

        self.assertEqual(state.deferral_count, 0)

    async def test_policy_disabled_preserves_legacy_ask_user_behaviour(self) -> None:
        """
        Backward-compat: turning off the policy reproduces the old behaviour
        (immediate ASK_USER on the first stuck signal).
        """

        state = AgentState(
            intent="x", capabilities=RuntimeCapabilities(hitl=HITLCapability(enabled=True))
        )
        state.set_sub_goals([SubGoalFixtures.make(description="v", index=0)])
        self.__seed_validate_only_loop(state=state)

        planner = StepPlanner(
            vision_tool=Mock(),
            escalation_policy=EscalationPolicy(enabled=False),
        )
        result = (
            await planner.plan_step(
                state=state,
                reasoner=Mock(),
                capture=self.__capture(),
                context_manager=self.__context(),
                screen_width=1080,
                screen_height=2400,
                prompt_if_stuck=True,
            )
        ).plan

        self.assertIsNotNone(result.step)
        assert result.step is not None
        self.assertIs(result.step.action.action_type, ActionType.ASK_USER)
        self.assertEqual(state.deferral_count, 0)
