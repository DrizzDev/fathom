from __future__ import annotations

import unittest

from fathom.core.agent.stuck_source import StuckSourceResolver
from fathom.schemas.escalation import StuckSource
from tests.builders import AgentFixtures, ScreenFixtures, SubGoalFixtures


class StuckSourceResolverTest(unittest.TestCase):
    """
    Pins the priority and None-return contract of :class:`StuckSourceResolver`.
    """

    def __resolver(self) -> StuckSourceResolver:
        """
        Construct the resolver under test.
        """

        return StuckSourceResolver()

    def test_returns_none_when_no_signal_active(self) -> None:
        """
        Fresh state with no signals returns None so callers can short-circuit.
        """

        state = AgentFixtures.state(intent="x")
        self.assertIsNone(self.__resolver().resolve(agent_state=state))

    def test_returns_loop_detector_when_only_loop_stuck(self) -> None:
        """
        Loop detector triggering without budget exhaustion resolves to LOOP_DETECTOR.
        """

        state = AgentFixtures.state(intent="x")
        detector = state.runtime.screen.detector
        screen = ScreenFixtures.state(activity="com.example/.Main", activity_hash="ah")
        for _ in range(detector.threshold):
            detector.record(screen=screen, action_type="tap", action_description="t")

        self.assertTrue(state.is_stuck)
        self.assertIs(self.__resolver().resolve(agent_state=state), StuckSource.LOOP_DETECTOR)

    def test_subgoal_budget_beats_loop_detector(self) -> None:
        """
        Budget exhaustion is a harder signal than the loop detector — priority.
        """

        state = AgentFixtures.state(intent="x")
        state.set_sub_goals([SubGoalFixtures.make(description="active", max_steps=1)])

        detector = state.runtime.screen.detector
        screen = ScreenFixtures.state(activity="com.example/.Main", activity_hash="ah")
        for _ in range(detector.threshold):
            detector.record(screen=screen, action_type="tap", action_description="t")
            state.record_sub_goal_action()

        self.assertTrue(state.is_stuck)
        self.assertTrue(state.current_sub_goal_over_budget)
        self.assertIs(self.__resolver().resolve(agent_state=state), StuckSource.SUBGOAL_BUDGET)

    def test_subgoal_budget_resolves_even_without_loop_signal(self) -> None:
        """
        Budget exhaustion is reachable independently of the loop detector.
        """

        state = AgentFixtures.state(intent="x")
        state.set_sub_goals([SubGoalFixtures.make(description="active", max_steps=1)])
        state.record_sub_goal_action()

        self.assertFalse(state.is_stuck)
        self.assertTrue(state.current_sub_goal_over_budget)
        self.assertIs(self.__resolver().resolve(agent_state=state), StuckSource.SUBGOAL_BUDGET)
