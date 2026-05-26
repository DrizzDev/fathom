"""
Unit pins for :class:`StepPlanner` escalation-gate wiring.

Covers both the deterministic ``is_stuck`` branch and the LLM-emitted
``ASK_USER`` gate, plus the deferral round-trip and ``should_retry``
contract that drives the GROUND -> ANALYZE re-plan loop.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from fathom.constants import ActionType
from fathom.constants.state import CompletionReason
from fathom.core.agent.planner import StepPlanner
from fathom.core.agent.state import AgentState
from fathom.schemas.actions import Action
from fathom.schemas.effect import ActionEffectStatus
from fathom.schemas.escalation import EscalationPolicy
from fathom.schemas.results import AnalysisResult
from fathom.schemas.screens import ScreenCapture, ScreenState
from fathom.schemas.subgoal import SubGoal


class StepPlannerEscalationTest(unittest.IsolatedAsyncioTestCase):
    """
    Pins escalation-gate behaviour exposed through :meth:`StepPlanner.plan_step`.
    """

    @staticmethod
    def __screen(*, visual_hash: str = "b" * 16) -> ScreenState:
        return ScreenState(
            activity="app",
            timestamp=0,
            activity_hash="a" * 16,
            visual_hash=visual_hash,
        )

    @staticmethod
    def __capture() -> ScreenCapture:
        return ScreenCapture(
            width=100,
            height=200,
            activity="app",
            image=b"png",
            timestamp=1,
        )

    @staticmethod
    def __context_manager() -> SimpleNamespace:
        """
        Minimal :class:`ContextManager` stand-in covering planner reads.

        ``inject_user_guidance`` is async because the production API is.
        """

        return SimpleNamespace(
            get_user_guidance=Mock(return_value=[]),
            inject_user_guidance=AsyncMock(),
            consume_user_guidance=Mock(),
            clear_user_guidance=Mock(),
            clear_verifier_feedback=Mock(),
        )

    def __validate_only_stuck_state(self) -> AgentState:
        """
        Build a state whose loop detector is stuck via passive-only NO_PROGRESS turns.
        """

        state = AgentState(intent="finish onboarding")
        state.set_sub_goals([SubGoal(description="Validate something", index=0, max_steps=10)])
        detector = state.runtime.screen.detector
        for _ in range(detector.threshold):
            detector.record(
                screen=self.__screen(),
                action_type="validate",
                action_description="validate srp",
                effect_status=ActionEffectStatus.NO_PROGRESS,
            )
        return state

    @staticmethod
    def __vision_with_navigation_action() -> Mock:
        """
        Vision stub that returns a benign TAP analysis so the fall-through path
        produces a definite planning result rather than awaiting a bare Mock.
        """

        action = Action(
            action_type=ActionType.TAP,
            target="Continue button",
            rationale="proceed",
            confidence=0.8,
        )
        analysis = AnalysisResult(
            action=action,
            reasoning="next step",
            screen_description="screen",
            metadata={"tool_args": {}},
        )
        vision = Mock()
        vision.analyze = AsyncMock(return_value=analysis)
        vision.build_rejection_history_from_analysis = Mock(return_value=[])
        return vision

    async def test_validate_only_stuck_defers_and_falls_through_to_analysis(
        self,
    ) -> None:
        """
        Two consecutive validate-only NO_PROGRESS turns defer escalation,
        bump the per-sub-goal deferral count, inject recovery guidance,
        and fall through so vision.analyze gets another chance.
        """

        state = self.__validate_only_stuck_state()
        context = self.__context_manager()
        vision = self.__vision_with_navigation_action()
        reasoner = Mock()
        reasoner.select_best_action.return_value = vision.analyze.return_value.action
        planner = StepPlanner(vision_tool=vision)

        await planner.plan_step(
            state=state,
            reasoner=reasoner,
            capture=self.__capture(),
            context_manager=context,
            screen_width=100,
            screen_height=200,
            interactive_mode=True,
            prompt_if_stuck=True,
        )

        self.assertEqual(state.deferral_count, 1)
        context.inject_user_guidance.assert_awaited()
        vision.analyze.assert_awaited()

    async def test_validate_only_at_tolerance_still_defers(self) -> None:
        """
        Three validate-only turns at tolerance=3 stay below the escalation cap.
        """

        state = AgentState(intent="x")
        state.set_sub_goals([SubGoal(description="v", index=0, max_steps=10)])
        detector = state.runtime.screen.detector
        for _ in range(3):
            detector.record(
                screen=self.__screen(),
                action_type="validate",
                action_description="v",
                effect_status=ActionEffectStatus.NO_PROGRESS,
            )

        vision = self.__vision_with_navigation_action()
        reasoner = Mock()
        reasoner.select_best_action.return_value = vision.analyze.return_value.action
        planner = StepPlanner(vision_tool=vision)
        context = self.__context_manager()
        await planner.plan_step(
            state=state,
            reasoner=reasoner,
            capture=self.__capture(),
            context_manager=context,
            screen_width=100,
            screen_height=200,
            interactive_mode=True,
            prompt_if_stuck=True,
        )
        self.assertEqual(state.deferral_count, 1)
        context.inject_user_guidance.assert_awaited()

    async def test_escape_valve_allows_ask_user_after_repeated_deferrals(self) -> None:
        """
        Once deferrals exceed the limit, the gate must escalate to ASK_USER.
        """

        state = self.__validate_only_stuck_state()
        # Drive the deferral count past the default limit of 2.
        state.record_deferral()
        state.record_deferral()
        state.record_deferral()

        planner = StepPlanner(vision_tool=Mock())
        result = await planner.plan_step(
            state=state,
            reasoner=Mock(),
            capture=self.__capture(),
            context_manager=self.__context_manager(),
            screen_width=100,
            screen_height=200,
            interactive_mode=True,
            prompt_if_stuck=True,
        )

        self.assertIsNotNone(result.step)
        assert result.step is not None
        self.assertIs(result.step.action.action_type, ActionType.ASK_USER)
        self.assertEqual(result.reason, CompletionReason.INTERVENTION_REQUIRED.value)
        self.assertEqual(state.deferral_count, 0)

    async def test_user_guidance_present_passes_through_without_gate(self) -> None:
        """
        Existing behaviour preserved: with active user guidance, the stuck
        branch passes through to analysis without invoking the gate.
        """

        state = self.__validate_only_stuck_state()
        context = self.__context_manager()
        context.get_user_guidance = Mock(return_value=[Mock(active=True)])

        vision = self.__vision_with_navigation_action()
        reasoner = Mock()
        reasoner.select_best_action.return_value = vision.analyze.return_value.action
        planner = StepPlanner(vision_tool=vision)
        await planner.plan_step(
            state=state,
            reasoner=reasoner,
            capture=self.__capture(),
            context_manager=context,
            screen_width=100,
            screen_height=200,
            interactive_mode=True,
            prompt_if_stuck=True,
        )
        # Gate was NOT invoked → deferral count unchanged → guidance not injected.
        self.assertEqual(state.deferral_count, 0)
        context.inject_user_guidance.assert_not_awaited()

    async def test_policy_disabled_preserves_original_ask_user_behaviour(self) -> None:
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
            capture=self.__capture(),
            context_manager=self.__context_manager(),
            screen_width=100,
            screen_height=200,
            interactive_mode=True,
            prompt_if_stuck=True,
        )

        self.assertIsNotNone(result.step)
        assert result.step is not None
        self.assertIs(result.step.action.action_type, ActionType.ASK_USER)
        self.assertEqual(state.deferral_count, 0)

    async def test_non_interactive_mode_skips_gate_entirely(self) -> None:
        """
        Autonomous mode keeps its existing recovery path; the gate must not run.
        """

        state = self.__validate_only_stuck_state()
        context = self.__context_manager()
        planner = StepPlanner(vision_tool=Mock())

        await planner.plan_step(
            state=state,
            reasoner=Mock(),
            capture=self.__capture(),
            context_manager=context,
            screen_width=100,
            screen_height=200,
            interactive_mode=False,
            prompt_if_stuck=False,
        )

        # Gate not invoked in non-interactive path.
        self.assertEqual(state.deferral_count, 0)
        context.inject_user_guidance.assert_not_awaited()


class StepPlannerLlmAskUserGateTest(unittest.IsolatedAsyncioTestCase):
    """
    Pins the LLM-emitted ``ASK_USER`` gate after :meth:`VisionService.analyze`.
    """

    @staticmethod
    def __capture() -> ScreenCapture:
        return ScreenCapture(
            width=100,
            height=200,
            activity="app",
            image=b"png",
            timestamp=1,
        )

    @staticmethod
    def __context_manager() -> SimpleNamespace:
        return SimpleNamespace(
            get_user_guidance=Mock(return_value=[]),
            inject_user_guidance=AsyncMock(),
            consume_user_guidance=Mock(),
            clear_user_guidance=Mock(),
            clear_verifier_feedback=Mock(),
        )

    @staticmethod
    def __ask_user_analysis() -> AnalysisResult:
        action = Action(
            action_type=ActionType.ASK_USER,
            target="user",
            rationale="missing credentials",
            confidence=0.9,
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
        state = AgentState(intent="x")
        state.set_sub_goals([SubGoal(description="v", index=0, max_steps=10)])
        detector = state.runtime.screen.detector
        for _ in range(detector.threshold):
            detector.record(
                screen=ScreenState(
                    activity="app",
                    timestamp=0,
                    activity_hash="ah",
                    visual_hash="b" * 16,
                ),
                action_type="validate",
                action_description="v",
                effect_status=ActionEffectStatus.NO_PROGRESS,
            )
        return state

    async def test_llm_ask_user_passes_through_when_no_stuck_source(self) -> None:
        """
        Fresh state with no stuck signal: the model's ASK_USER tool call is
        a legitimate request for external information; let it through.
        """

        analysis = self.__ask_user_analysis()
        vision = Mock()
        vision.analyze = AsyncMock(return_value=analysis)
        vision.build_rejection_history_from_analysis = Mock(return_value=[])

        state = AgentState(intent="login")
        state.set_sub_goals([SubGoal(description="login", index=0)])
        reasoner = Mock()
        reasoner.select_best_action.return_value = analysis.action

        planner = StepPlanner(vision_tool=vision)
        result = await planner.plan_step(
            state=state,
            reasoner=reasoner,
            capture=self.__capture(),
            context_manager=self.__context_manager(),
            screen_width=100,
            screen_height=200,
            interactive_mode=False,
            prompt_if_stuck=False,
        )

        self.assertIsNotNone(result.step)
        assert result.step is not None
        self.assertIs(result.step.action.action_type, ActionType.ASK_USER)
        self.assertFalse(result.should_retry)
        self.assertEqual(state.deferral_count, 0)

    async def test_llm_ask_user_deferred_when_validate_only_loop_active(self) -> None:
        """
        With an active validate-only loop signal, an LLM-emitted ASK_USER is
        deferred via ``should_retry=True`` and the deferral count increments.

        Uses ``interactive_mode=True`` with active user guidance so the
        deterministic stuck branch passes through (the planner-synthesized
        path is bypassed) and we land on ``vision.analyze`` directly.
        """

        analysis = self.__ask_user_analysis()
        vision = Mock()
        vision.analyze = AsyncMock(return_value=analysis)
        vision.build_rejection_history_from_analysis = Mock(return_value=[])

        state = self.__validate_only_stuck_state()
        reasoner = Mock()
        reasoner.select_best_action.return_value = analysis.action

        planner = StepPlanner(vision_tool=vision)
        context = self.__context_manager()
        context.get_user_guidance = Mock(return_value=[Mock(active=True)])
        result = await planner.plan_step(
            state=state,
            reasoner=reasoner,
            capture=self.__capture(),
            context_manager=context,
            screen_width=100,
            screen_height=200,
            interactive_mode=True,
            prompt_if_stuck=True,
        )

        self.assertIsNone(result.step)
        self.assertTrue(result.should_retry)
        self.assertFalse(result.is_complete)
        self.assertEqual(state.deferral_count, 1)
        context.inject_user_guidance.assert_awaited()
        self.assertEqual(result.metadata.get("escalation.path"), "llm_tool")
        self.assertTrue(result.metadata.get("escalation.suppressed"))
