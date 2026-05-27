"""
Release-gated wiring smoke test for the escalation gate.

Verifies that :class:`IntentConfiguration` values flow end-to-end into
:class:`StepPlanner` so a custom :class:`EscalationPolicy` configured via
``FathomConfiguration`` is actually honoured at decision time. This is the
"deployed integration" version of the unit tests — it constructs the real
configuration object graph rather than instantiating the gate directly.

No LLM call. Gated under ``pytest.mark.release`` so it runs alongside the
other live-suite tests in staging release pipelines.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from fathom.constants import ActionType
from fathom.core.agent.planner import StepPlanner
from fathom.core.agent.state import AgentState
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
        return ScreenState(
            activity="com.example/.Main",
            timestamp=0,
            activity_hash="ah",
            visual_hash="b" * 16,
        )

    @staticmethod
    def __capture() -> ScreenCapture:
        return ScreenCapture(
            width=1080,
            height=2400,
            activity="com.example/.Main",
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
    def __validate_only_stuck_state() -> AgentState:
        state = AgentState(
            intent="x", capabilities=RuntimeCapabilities(hitl=HITLCapability(enabled=True))
        )
        state.set_sub_goals([SubGoal(description="v", index=0, max_steps=10)])
        detector = state.runtime.screen.detector
        for _ in range(detector.threshold):
            detector.record(
                screen=EscalationGateWiringTest.__screen(),
                action_type="validate",
                action_description="v",
                effect_status=ActionEffectStatus.NO_PROGRESS,
            )
        return state

    async def test_intent_configuration_carries_escalation_policy_by_default(self) -> None:
        """
        Newly constructed :class:`IntentConfiguration` has the default policy attached.
        """

        config = IntentConfiguration()
        self.assertIsInstance(config.escalation, EscalationPolicy)
        self.assertTrue(config.escalation.enabled)
        self.assertEqual(config.escalation.deferral_limit, 2)
        self.assertEqual(config.escalation.passive_tolerance, 3)

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
        result = await planner.plan_step(
            state=state,
            reasoner=Mock(),
            capture=self.__capture(),
            context_manager=self.__context(),
            screen_width=1080,
            screen_height=2400,
            prompt_if_stuck=True,
        )

        # Two NO_PROGRESS validates with tolerance=1 → escalate.
        self.assertIsNotNone(result.step)
        assert result.step is not None
        self.assertIs(result.step.action.action_type, ActionType.ASK_USER)
        self.assertEqual(state.deferral_count, 0)

    async def test_planner_honours_disabled_policy_from_configuration(self) -> None:
        """
        When the configuration disables the gate the planner reproduces the
        original ASK_USER-on-stuck behaviour.
        """

        config = IntentConfiguration(
            escalation=EscalationPolicy(enabled=False),
        )
        planner = StepPlanner(
            vision_tool=Mock(),
            escalation_policy=config.escalation,
        )

        state = self.__validate_only_stuck_state()
        result = await planner.plan_step(
            state=state,
            reasoner=Mock(),
            capture=self.__capture(),
            context_manager=self.__context(),
            screen_width=1080,
            screen_height=2400,
            prompt_if_stuck=True,
        )

        self.assertIsNotNone(result.step)
        assert result.step is not None
        self.assertIs(result.step.action.action_type, ActionType.ASK_USER)
