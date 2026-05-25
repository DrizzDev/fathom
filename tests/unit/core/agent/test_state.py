from __future__ import annotations

import unittest

from fathom.core.agent.state import AgentState
from fathom.schemas.screens import ScreenState


class AgentStateContinuationTest(unittest.TestCase):
    """
    Covers continuation policy across autonomous and HITL modes.
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

    def test_interactive_mode_can_continue_after_autonomous_recovery_exhausted(self) -> None:
        """
        HITL mode is governed by the realignment budget, not the native
        autonomous recovery budget.
        """

        state = AgentState(intent="complete onboarding")
        detector = state.runtime.screen.detector
        for _ in range(detector.threshold):
            detector.record(
                screen=self.__screen(),
                action_type="tap",
                action_description="Tap Continue",
            )

        self.assertTrue(state.is_stuck)
        while detector.can_recover():
            detector.record_recovery_attempt()

        self.assertFalse(state.can_continue_with(interactive_mode=False))
        self.assertTrue(state.can_continue_with(interactive_mode=True))

    def test_interactive_mode_stops_when_realignment_budget_is_exhausted(self) -> None:
        """
        HITL mode terminates once the realignment budget is exhausted.
        """

        state = AgentState(intent="complete onboarding", realignment_budget=1)
        detector = state.runtime.screen.detector
        for _ in range(detector.threshold):
            detector.record(
                screen=self.__screen(),
                action_type="tap",
                action_description="Tap Continue",
            )

        state.bump_realignment_budget()

        self.assertTrue(state.is_stuck)
        self.assertFalse(state.can_continue_with(interactive_mode=True))
