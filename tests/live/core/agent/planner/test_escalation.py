from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, Mock

import pytest

from fathom.constants import ActionType
from fathom.core.agent.planner import StepPlanner
from fathom.core.agent.state import AgentState
from fathom.core.context.manager import ContextManager
from fathom.schemas.capabilities import HITLCapability, RuntimeCapabilities
from fathom.schemas.configuration import IntentConfiguration
from fathom.schemas.effect import ActionEffectStatus
from fathom.schemas.escalation import EscalationPolicy
from fathom.schemas.screens import ScreenCapture, ScreenState
from fathom.schemas.subgoal import SubGoal

pytestmark = pytest.mark.release


class EscalationGateWiringTest(unittest.IsolatedAsyncioTestCase):
    """
    Pins configuration-to-planner wiring for the escalation gate.
    """

    @staticmethod
    def __screen() -> ScreenState:
        """ """

        return ScreenState(
            timestamp=0,
            activity_hash="ah",
            visual_hash="b" * 16,
            activity="com.example/.Main",
        )

    @staticmethod
    def __capture() -> ScreenCapture:
        """ """

        return ScreenCapture(
            width=1080,
            height=2400,
            image=b"png",
            timestamp=1,
            activity="com.example/.Main",
        )

    @staticmethod
    def __context() -> ContextManager:
        """
        Build a typed :class:`ContextManager` stub for planner.plan_step tests.
        """

        context = Mock(spec=ContextManager)

        context.clear_user_guidance = Mock()
        context.consume_user_guidance = Mock()
        context.clear_verifier_feedback = Mock()
        context.inject_user_guidance = AsyncMock()
        context.get_user_guidance = Mock(return_value=[])

        return context

    @staticmethod
    def __validate_only_stuck_state() -> AgentState:
        """ """

        state = AgentState(
            intent="x", capabilities=RuntimeCapabilities(hitl=HITLCapability(enabled=True))
        )
        state.set_sub_goals([SubGoal(description="v", index=0, max_steps=10)])
        detector = state.runtime.screen.detector

        for _ in range(detector.threshold):
            detector.record(
                action_type="validate",
                action_description="v",
                screen=EscalationGateWiringTest.__screen(),
                effect_status=ActionEffectStatus.NO_PROGRESS,
            )

        return state

    async def test_intent_configuration_carries_escalation_policy_by_default(self) -> None:
        """
        Newly constructed :class:`IntentConfiguration` has the default policy attached.
        """

        config = IntentConfiguration()

        self.assertTrue(config.escalation.enabled)
        self.assertEqual(config.escalation.deferral_limit, 2)
        self.assertEqual(config.escalation.passive_tolerance, 3)
        self.assertIsInstance(config.escalation, EscalationPolicy)

    async def test_planner_honours_custom_policy_from_configuration(self) -> None:
        """
        A custom policy with ``passive_tolerance=1`` must escalate after the
        second validate-only turn instead of deferring.
        """

        config = IntentConfiguration(
            escalation=EscalationPolicy(passive_tolerance=1),
        )
        planner = StepPlanner(
            vision_tool=Mock(),
            escalation_policy=config.escalation,
        )

        state = self.__validate_only_stuck_state()
        result = (
            await planner.plan_step(
                state=state,
                reasoner=Mock(),
                screen_width=1080,
                screen_height=2400,
                prompt_if_stuck=True,
                capture=self.__capture(),
                context_manager=self.__context(),
            )
        ).plan

        # Two NO_PROGRESS validates with tolerance=1 → escalate.
        self.assertIsNotNone(result.step)
        assert result.step is not None

        self.assertEqual(state.deferral_count, 0)
        self.assertIs(result.step.action.action_type, ActionType.ASK_USER)

    async def test_planner_honours_disabled_policy_from_configuration(self) -> None:
        """
        When the configuration disables the gate the planner reproduces the original ASK_USER-on-stuck behavior.
        """

        config = IntentConfiguration(
            escalation=EscalationPolicy(enabled=False),
        )
        planner = StepPlanner(
            vision_tool=Mock(),
            escalation_policy=config.escalation,
        )

        state = self.__validate_only_stuck_state()
        result = (
            await planner.plan_step(
                state=state,
                reasoner=Mock(),
                screen_width=1080,
                screen_height=2400,
                prompt_if_stuck=True,
                capture=self.__capture(),
                context_manager=self.__context(),
            )
        ).plan

        self.assertIsNotNone(result.step)
        assert result.step is not None
        self.assertIs(result.step.action.action_type, ActionType.ASK_USER)
