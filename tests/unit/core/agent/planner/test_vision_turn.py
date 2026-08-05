from __future__ import annotations

import unittest

from fathom.core.agent.planner.vision_turn import VisionTurn
from fathom.core.agent.state import AgentState
from fathom.schemas.capabilities import HITLCapability, RuntimeCapabilities
from tests.builders import SubGoalFixtures
from tests.builders.success import SuccessFixtures


class VisionTurnAssertionThreadingTest(unittest.TestCase):
    """
    Pins that the active observed assertion is threaded into the vision sub-goal context.
    """

    @staticmethod
    def __caps() -> RuntimeCapabilities:
        """
        Autonomous capabilities so no HITL gating interferes with the cursor.
        """

        return RuntimeCapabilities(hitl=HITLCapability(enabled=False))

    @staticmethod
    def __context(*, state: AgentState) -> object:
        """
        Invoke the private sub-goal-context mapping under test.
        """

        return VisionTurn._VisionTurn__sub_goal_context(state=state)  # type: ignore[attr-defined]

    def test_observed_sub_goal_threads_its_assertion(self) -> None:
        """
        An active observed sub-goal exposes its exact assertion as the completion condition.
        """

        state = AgentState(intent="buy ghar soap", capabilities=self.__caps())
        state.set_sub_goals(
            [
                SubGoalFixtures.make(
                    index=0,
                    description="Open Amazon and search",
                    success=SuccessFixtures.observed(
                        assertion="The Amazon search results for 'ghar soap' are displayed."
                    ),
                ),
                SubGoalFixtures.make(index=1, description="Confirm"),
            ]
        )

        context = self.__context(state=state)

        assert context is not None
        self.assertEqual(
            context["assertion"], "The Amazon search results for 'ghar soap' are displayed."
        )
        self.assertFalse(context["durable"])

    def test_command_sub_goal_threads_no_assertion(self) -> None:
        """
        A command sub-goal carries no observation, so no completion-condition assertion is threaded.
        """

        state = AgentState(intent="buy ghar soap", capabilities=self.__caps())
        state.set_sub_goals(
            [
                SubGoalFixtures.make(
                    index=0,
                    description="Tap the search field",
                    success=SuccessFixtures.command(quote="tap", intent="tap the target"),
                ),
                SubGoalFixtures.make(index=1, description="Confirm"),
            ]
        )

        context = self.__context(state=state)

        assert context is not None
        self.assertNotIn("assertion", context)
        self.assertTrue(context["durable"])
