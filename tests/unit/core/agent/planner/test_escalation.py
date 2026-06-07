from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, Mock

from tests.builders import ActionFixtures, AgentFixtures, ScreenFixtures, SubGoalFixtures

from fathom.constants import ActionType
from fathom.constants.state import CompletionReason
from fathom.core.agent.planner import StepPlanner
from fathom.core.agent.state import AgentState
from fathom.schemas.capabilities import HITLCapability, RuntimeCapabilities
from fathom.schemas.effect import ActionEffectStatus
from fathom.schemas.escalation import EscalationPolicy
from fathom.schemas.results import AnalysisResult


class StepPlannerEscalationTest(unittest.IsolatedAsyncioTestCase):
    """
    Pins escalation-gate behavior exposed through :meth:`StepPlanner.plan_step`.
    """

    def __validate_only_stuck_state(
        self,
        *,
        capabilities: RuntimeCapabilities = RuntimeCapabilities(hitl=HITLCapability(enabled=True)),
    ) -> AgentState:
        """
        Build a state whose loop detector is stuck via passive-only NO_PROGRESS turns.
        """

        state = AgentState(intent="finish onboarding", capabilities=capabilities)
        state.set_sub_goals([SubGoalFixtures.make(description="Validate something", max_steps=10)])

        detector = state.runtime.screen.detector
        screen = ScreenFixtures.state(activity="app")
        for _ in range(detector.threshold):
            detector.record(
                screen=screen,
                action_type="validate",
                action_description="validate srp",
                effect_status=ActionEffectStatus.NO_PROGRESS,
            )

        return state

    @staticmethod
    def __vision_with_navigation_action() -> Mock:
        """
        Vision stub returning a benign TAP analysis so the fall-through path
        produces a definite planning result.
        """

        action = ActionFixtures.tap(
            target="Continue button",
            rationale="proceed",
            confidence=0.8,
        )
        analysis = AnalysisResult(
            action=action,
            reasoning="next step",
            metadata={"tool_args": {}},
            screen_description="screen",
        )
        vision = Mock()
        vision.analyze = AsyncMock(return_value=analysis)
        vision.build_rejection_history_from_analysis = Mock(return_value=[])
        return vision

    async def test_validate_only_stuck_defers_and_falls_through_to_analysis(
        self,
    ) -> None:
        """
        Two consecutive validate-only NO_PROGRESS turns defer escalation, bump
        the per-sub-goal deferral count, inject recovery guidance, and fall
        through so vision.analyze gets another chance.
        """

        context = AgentFixtures.context_manager_stub()
        vision = self.__vision_with_navigation_action()
        state = self.__validate_only_stuck_state()

        reasoner = Mock()
        planner = StepPlanner(vision_tool=vision)
        reasoner.select_best_action.return_value = vision.analyze.return_value.action

        await planner.plan_step(
            state=state,
            screen_width=100,
            screen_height=200,
            reasoner=reasoner,
            prompt_if_stuck=True,
            context_manager=context,
            capture=ScreenFixtures.capture(activity="app"),
        )

        self.assertEqual(state.deferral_count, 1)
        context.inject_user_guidance.assert_awaited()
        vision.analyze.assert_awaited()

    async def test_validate_only_at_tolerance_still_defers(self) -> None:
        """
        Three validate-only turns at tolerance=3 stay below the escalation cap.
        """

        state = AgentFixtures.state(intent="x", hitl_enabled=True)
        state.set_sub_goals([SubGoalFixtures.make(description="v", max_steps=10)])
        detector = state.runtime.screen.detector
        screen = ScreenFixtures.state(activity="app")
        for _ in range(3):
            detector.record(
                screen=screen,
                action_type="validate",
                action_description="v",
                effect_status=ActionEffectStatus.NO_PROGRESS,
            )

        vision = self.__vision_with_navigation_action()

        reasoner = Mock()
        reasoner.select_best_action.return_value = vision.analyze.return_value.action
        planner = StepPlanner(vision_tool=vision)

        context = AgentFixtures.context_manager_stub()
        await planner.plan_step(
            state=state,
            screen_width=100,
            screen_height=200,
            reasoner=reasoner,
            prompt_if_stuck=True,
            context_manager=context,
            capture=ScreenFixtures.capture(activity="app"),
        )
        self.assertEqual(state.deferral_count, 1)
        context.inject_user_guidance.assert_awaited()

    async def test_escape_valve_allows_ask_user_after_repeated_deferrals(self) -> None:
        """
        Once deferrals exceed the limit, the gate must escalate to ASK_USER.
        """

        state = self.__validate_only_stuck_state()
        state.record_deferral()
        state.record_deferral()
        state.record_deferral()

        planner = StepPlanner(vision_tool=Mock())
        result = await planner.plan_step(
            state=state,
            reasoner=Mock(),
            screen_width=100,
            screen_height=200,
            prompt_if_stuck=True,
            capture=ScreenFixtures.capture(activity="app"),
            context_manager=AgentFixtures.context_manager_stub(),
        )

        self.assertIsNotNone(result.step)
        assert result.step is not None

        self.assertEqual(state.deferral_count, 0)
        self.assertIs(result.step.action.action_type, ActionType.ASK_USER)
        self.assertEqual(result.reason, CompletionReason.INTERVENTION_REQUIRED.value)

    async def test_user_guidance_present_passes_through_without_gate(self) -> None:
        """
        With active user guidance, the stuck branch passes through to analysis
        without invoking the gate.
        """

        context = AgentFixtures.context_manager_stub(user_guidance=[Mock(active=True)])
        state = self.__validate_only_stuck_state()

        vision = self.__vision_with_navigation_action()

        reasoner = Mock()
        reasoner.select_best_action.return_value = vision.analyze.return_value.action
        planner = StepPlanner(vision_tool=vision)

        await planner.plan_step(
            state=state,
            screen_width=100,
            screen_height=200,
            reasoner=reasoner,
            prompt_if_stuck=True,
            context_manager=context,
            capture=ScreenFixtures.capture(activity="app"),
        )
        self.assertEqual(state.deferral_count, 0)
        context.inject_user_guidance.assert_not_awaited()

    async def test_policy_disabled_preserves_original_ask_user_behavior(self) -> None:
        """
        When the policy is disabled the planner returns ASK_USER without deferral.
        """

        state = self.__validate_only_stuck_state()
        planner = StepPlanner(
            vision_tool=Mock(),
            escalation_policy=EscalationPolicy(enabled=False),
        )
        result = await planner.plan_step(
            state=state,
            reasoner=Mock(),
            screen_width=100,
            screen_height=200,
            prompt_if_stuck=True,
            capture=ScreenFixtures.capture(activity="app"),
            context_manager=AgentFixtures.context_manager_stub(),
        )

        self.assertIsNotNone(result.step)
        assert result.step is not None

        self.assertEqual(state.deferral_count, 0)
        self.assertIs(result.step.action.action_type, ActionType.ASK_USER)

    async def test_autonomous_capability_skips_gate_entirely(self) -> None:
        """
        Autonomous runtime skips the HITL gate; validate-passive stuck falls
        through to vision re-planning.
        """

        state = self.__validate_only_stuck_state(
            capabilities=RuntimeCapabilities(hitl=HITLCapability(enabled=False))
        )

        context = AgentFixtures.context_manager_stub()
        vision = self.__vision_with_navigation_action()

        reasoner = Mock()
        reasoner.select_best_action.return_value = vision.analyze.return_value.action
        planner = StepPlanner(vision_tool=vision)

        await planner.plan_step(
            state=state,
            screen_width=100,
            screen_height=200,
            reasoner=reasoner,
            prompt_if_stuck=False,
            context_manager=context,
            capture=ScreenFixtures.capture(activity="app"),
        )

        self.assertEqual(state.deferral_count, 0)
        context.inject_user_guidance.assert_not_awaited()


class StepPlannerLlmAskUserGateTest(unittest.IsolatedAsyncioTestCase):
    """
    Pins the LLM-emitted ``ASK_USER`` gate after :meth:`VisionService.analyze`.
    """

    @staticmethod
    def __ask_user_analysis() -> AnalysisResult:
        """
        Build an ``AnalysisResult`` whose action is ``ASK_USER``.
        """

        action = ActionFixtures.make(
            target="user",
            action_type=ActionType.ASK_USER,
            rationale="missing credentials",
            text="What is your password?",
        )
        return AnalysisResult(
            action=action,
            reasoning="Need credentials",
            screen_description="login",
            metadata={"tool_args": {}},
        )

    @staticmethod
    def __validate_only_stuck_state() -> AgentState:
        """
        Build a stuck state whose only repeating action is a validation step.
        """

        state = AgentFixtures.state(intent="x", hitl_enabled=True)
        state.set_sub_goals([SubGoalFixtures.make(description="v", max_steps=10)])
        detector = state.runtime.screen.detector
        screen = ScreenFixtures.state(activity="app", activity_hash="ah")
        for _ in range(detector.threshold):
            detector.record(
                screen=screen,
                action_type="validate",
                action_description="v",
                effect_status=ActionEffectStatus.NO_PROGRESS,
            )
        return state

    async def test_llm_ask_user_passes_through_when_no_stuck_source(self) -> None:
        """
        Fresh interactive state with no stuck signal: the model's ASK_USER
        tool call is a legitimate request for external information.
        """

        analysis = self.__ask_user_analysis()

        vision = Mock()
        vision.analyze = AsyncMock(return_value=analysis)
        vision.build_rejection_history_from_analysis = Mock(return_value=[])

        state = AgentFixtures.state(intent="login", hitl_enabled=True)
        state.set_sub_goals([SubGoalFixtures.make(description="login")])

        reasoner = Mock()
        reasoner.select_best_action.return_value = analysis.action

        planner = StepPlanner(vision_tool=vision)
        result = await planner.plan_step(
            state=state,
            screen_width=100,
            screen_height=200,
            reasoner=reasoner,
            prompt_if_stuck=False,
            capture=ScreenFixtures.capture(activity="app"),
            context_manager=AgentFixtures.context_manager_stub(),
        )

        self.assertIsNotNone(result.step)
        assert result.step is not None

        self.assertFalse(result.should_retry)
        self.assertEqual(state.deferral_count, 0)
        self.assertIs(result.step.action.action_type, ActionType.ASK_USER)

    async def test_llm_ask_user_deferred_when_validate_only_loop_active(self) -> None:
        """
        With an active validate-only loop signal the LLM-emitted ASK_USER is
        deferred (``should_retry=True``) and the deferral count increments.
        """

        analysis = self.__ask_user_analysis()

        vision = Mock()
        vision.analyze = AsyncMock(return_value=analysis)
        vision.build_rejection_history_from_analysis = Mock(return_value=[])

        state = self.__validate_only_stuck_state()

        reasoner = Mock()
        reasoner.select_best_action.return_value = analysis.action

        planner = StepPlanner(vision_tool=vision)
        context = AgentFixtures.context_manager_stub(user_guidance=[Mock(active=True)])

        result = await planner.plan_step(
            state=state,
            screen_width=100,
            screen_height=200,
            reasoner=reasoner,
            prompt_if_stuck=True,
            context_manager=context,
            capture=ScreenFixtures.capture(activity="app"),
        )

        self.assertIsNone(result.step)
        self.assertTrue(result.should_retry)
        self.assertFalse(result.is_complete)
        self.assertEqual(state.deferral_count, 1)

        context.inject_user_guidance.assert_awaited()

        self.assertTrue(result.metadata.get("escalation.suppressed"))
        self.assertEqual(result.metadata.get("escalation.path"), "llm_tool")
