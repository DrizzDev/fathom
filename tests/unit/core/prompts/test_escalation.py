from __future__ import annotations

import unittest

from fathom.core.prompts.escalation import EscalationPromptBuilder
from fathom.schemas.escalation import StuckSource
from fathom.schemas.subgoal import SubGoal


class EscalationPromptBuilderTest(unittest.TestCase):
    """
    Pins the rationale and question grammar of :class:`EscalationPromptBuilder`.
    """

    def test_loop_detector_emits_loop_rationale(self) -> None:
        """
        LOOP_DETECTOR source picks the loop-specific rationale string and a neutral question.
        """

        prompt = EscalationPromptBuilder.build(
            source=StuckSource.LOOP_DETECTOR,
            current_sub_goal=None,
            last_action_description=None,
        )

        self.assertIn("Loop detected", prompt.rationale)
        self.assertNotIn("budget", prompt.question.lower())
        self.assertNotIn("sub-goal", prompt.question.lower())

    def test_subgoal_budget_emits_budget_rationale(self) -> None:
        """
        SUBGOAL_BUDGET source picks the budget-specific rationale; question stays neutral.
        """

        prompt = EscalationPromptBuilder.build(
            source=StuckSource.SUBGOAL_BUDGET,
            current_sub_goal=None,
            last_action_description=None,
        )

        self.assertIn("budget", prompt.rationale.lower())
        self.assertNotIn("sub-goal", prompt.question.lower())
        self.assertNotIn("step budget", prompt.question.lower())

    def test_sub_goal_context_appears_in_question(self) -> None:
        """
        The active sub-goal description must surface in the user-facing question.
        """

        sub_goal = SubGoal(description="Press the Play button", index=0)
        prompt = EscalationPromptBuilder.build(
            current_sub_goal=sub_goal,
            last_action_description=None,
            source=StuckSource.LOOP_DETECTOR,
        )

        self.assertIn("Press the Play button", prompt.question)

    def test_last_action_descriptor_appears_in_question(self) -> None:
        """
        The repeated action descriptor must surface in the user-facing question.
        """

        prompt = EscalationPromptBuilder.build(
            current_sub_goal=None,
            source=StuckSource.LOOP_DETECTOR,
            last_action_description="Tap on Play button",
        )

        self.assertIn("Tap on Play button", prompt.question)

    def test_missing_context_does_not_break_question(self) -> None:
        """
        The question must remain grammatically reasonable when context is absent.
        """

        prompt = EscalationPromptBuilder.build(
            current_sub_goal=None,
            last_action_description=None,
            source=StuckSource.LOOP_DETECTOR,
        )

        self.assertNotIn("None", prompt.question)
        self.assertTrue(prompt.question.endswith("?"))


if __name__ == "__main__":
    unittest.main()
