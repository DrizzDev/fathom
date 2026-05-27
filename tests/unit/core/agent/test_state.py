from __future__ import annotations

import unittest

from fathom.core.agent.state import AgentState
from fathom.schemas.capabilities import HITLCapability, RuntimeCapabilities
from fathom.schemas.screens import ScreenState


class AgentStateContinuationTest(unittest.TestCase):
    """
    Covers continuation policy across autonomous and HITL runtimes via state.capabilities.
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

    def test_hitl_capability_can_continue_after_autonomous_recovery_exhausted(self) -> None:
        """
        HITL-capable runtime is governed by the realignment budget, not the
        autonomous recovery budget. The capability flag on AgentState picks
        the correct stuck-rescue budget.
        """

        autonomous = AgentState(
            intent="complete onboarding",
            capabilities=RuntimeCapabilities(hitl=HITLCapability(enabled=False)),
        )
        detector = autonomous.runtime.screen.detector
        for _ in range(detector.threshold):
            detector.record(
                screen=self.__screen(),
                action_type="tap",
                action_description="Tap Continue",
            )

        self.assertTrue(autonomous.is_stuck)
        while detector.can_recover():
            detector.record_recovery_attempt()

        self.assertFalse(autonomous.can_continue)

        hitl = AgentState(
            intent="complete onboarding",
            capabilities=RuntimeCapabilities(hitl=HITLCapability(enabled=True)),
        )
        hitl_detector = hitl.runtime.screen.detector
        for _ in range(hitl_detector.threshold):
            hitl_detector.record(
                screen=self.__screen(),
                action_type="tap",
                action_description="Tap Continue",
            )
        while hitl_detector.can_recover():
            hitl_detector.record_recovery_attempt()

        self.assertTrue(hitl.can_continue)

    def test_from_checkpoint_uses_live_capabilities_not_stale(self) -> None:
        """Restoration must bind capabilities from the resuming runtime, not the checkpoint."""

        original = AgentState(
            intent="x",
            capabilities=RuntimeCapabilities(hitl=HITLCapability(enabled=True)),
            max_steps=7,
        )
        original.mark_complete(reason="done")
        payload = original.to_checkpoint()

        self.assertNotIn("hitl", payload)
        self.assertNotIn("capabilities", payload)

        restored = AgentState.from_checkpoint(
            payload,
            capabilities=RuntimeCapabilities(hitl=HITLCapability(enabled=False)),
        )

        self.assertFalse(restored.capabilities.hitl.enabled)
        self.assertEqual(restored.intent, "x")
        self.assertTrue(restored.is_complete)
        self.assertEqual(restored.completion_reason, "done")

    def test_from_checkpoint_requires_capabilities_argument(self) -> None:
        """from_checkpoint must reject a missing capabilities kwarg fail-fast."""

        payload = AgentState(
            intent="x",
            capabilities=RuntimeCapabilities(hitl=HITLCapability(enabled=False)),
        ).to_checkpoint()

        with self.assertRaises(TypeError):
            AgentState.from_checkpoint(payload)  # type: ignore[call-arg]

    def test_hitl_capability_stops_when_realignment_budget_is_exhausted(self) -> None:
        """
        HITL-capable runtime terminates once the realignment budget is exhausted.
        """

        state = AgentState(
            intent="complete onboarding",
            capabilities=RuntimeCapabilities(hitl=HITLCapability(enabled=True)),
            realignment_budget=1,
        )
        detector = state.runtime.screen.detector
        for _ in range(detector.threshold):
            detector.record(
                screen=self.__screen(),
                action_type="tap",
                action_description="Tap Continue",
            )

        state.bump_realignment_budget()

        self.assertTrue(state.is_stuck)
        self.assertFalse(state.can_continue)
