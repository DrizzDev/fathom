"""
Unit pins for :class:`StuckSourceResolver` priority and absence semantics.
"""

from __future__ import annotations

import unittest

from fathom.core.agent.state import AgentState
from fathom.core.agent.stuck_source import StuckSourceResolver
from fathom.schemas.capabilities import HITLCapability, RuntimeCapabilities
from fathom.schemas.screens import ScreenState
from fathom.schemas.subgoal import SubGoal


class StuckSourceResolverTest(unittest.TestCase):
    """
    Pins the priority and None-return contract of :class:`StuckSourceResolver`.
    """

    @staticmethod
    def __screen() -> ScreenState:
        return ScreenState(
            activity="com.example/.Main",
            timestamp=0,
            activity_hash="ah",
            visual_hash="b" * 16,
        )

    def __resolver(self) -> StuckSourceResolver:
        return StuckSourceResolver()

    def test_returns_none_when_no_signal_active(self) -> None:
        """
        Fresh state with no signals returns None so callers can short-circuit.
        """

        state = AgentState(
            intent="x", capabilities=RuntimeCapabilities(hitl=HITLCapability(enabled=False))
        )
        self.assertIsNone(self.__resolver().resolve(agent_state=state))

    def test_returns_loop_detector_when_only_loop_stuck(self) -> None:
        """
        Loop detector triggering without budget exhaustion resolves to LOOP_DETECTOR.
        """

        from fathom.schemas.escalation import StuckSource

        state = AgentState(
            intent="x", capabilities=RuntimeCapabilities(hitl=HITLCapability(enabled=False))
        )
        detector = state.runtime.screen.detector
        for _ in range(detector.threshold):
            detector.record(
                screen=self.__screen(),
                action_type="tap",
                action_description="t",
            )

        self.assertTrue(state.is_stuck)
        self.assertIs(self.__resolver().resolve(agent_state=state), StuckSource.LOOP_DETECTOR)

    def test_subgoal_budget_beats_loop_detector(self) -> None:
        """
        Budget exhaustion is a harder signal than the loop detector — priority.
        """

        from fathom.schemas.escalation import StuckSource

        state = AgentState(
            intent="x", capabilities=RuntimeCapabilities(hitl=HITLCapability(enabled=False))
        )
        state.set_sub_goals([SubGoal(description="active", index=0, max_steps=1)])

        # Trip both signals simultaneously: budget exhausted + loop stuck.
        detector = state.runtime.screen.detector
        for _ in range(detector.threshold):
            detector.record(
                screen=self.__screen(),
                action_type="tap",
                action_description="t",
            )
            state.record_sub_goal_action()

        self.assertTrue(state.is_stuck)
        self.assertTrue(state.current_sub_goal_over_budget)
        self.assertIs(self.__resolver().resolve(agent_state=state), StuckSource.SUBGOAL_BUDGET)

    def test_subgoal_budget_resolves_even_without_loop_signal(self) -> None:
        """
        Budget exhaustion is reachable independently of the loop detector.
        """

        from fathom.schemas.escalation import StuckSource

        state = AgentState(
            intent="x", capabilities=RuntimeCapabilities(hitl=HITLCapability(enabled=False))
        )
        state.set_sub_goals([SubGoal(description="active", index=0, max_steps=1)])
        state.record_sub_goal_action()

        self.assertFalse(state.is_stuck)
        self.assertTrue(state.current_sub_goal_over_budget)
        self.assertIs(self.__resolver().resolve(agent_state=state), StuckSource.SUBGOAL_BUDGET)
