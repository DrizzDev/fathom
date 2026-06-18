from __future__ import annotations

import unittest

from pydantic import ValidationError

from fathom.constants import ActionType
from fathom.constants.exploration import BFSPhase, ExpectedOutcome
from fathom.schemas.actions import Action
from fathom.schemas.exploration import ActionOutcome, ExplorationProgress, TokenUsage
from fathom.schemas.steps import Step, StepResult


class TestActionOutcome(unittest.TestCase):
    """ActionOutcome projects an executed step result into scan feedback."""

    @staticmethod
    def __result(
        *, action: Action, success: bool = True, screen_changed: bool = True
    ) -> StepResult:
        step = Step(action=action, screen_hash="pre", step_number=1)
        return StepResult(
            step=step,
            success=success,
            duration=5,
            screen_changed=screen_changed,
            pre_hash="pre",
            post_hash="post",
        )

    def test_projects_action_type_and_label(self) -> None:
        action = Action(
            action_type=ActionType.TAP, rationale="r", natural_language_target="Home tab"
        )

        outcome = ActionOutcome.from_step_result(result=self.__result(action=action))

        self.assertEqual(outcome.kind, ActionType.TAP)
        self.assertEqual(outcome.target, "Home tab")
        self.assertTrue(outcome.success)
        self.assertTrue(outcome.screen_changed)

    def test_falls_back_to_target_when_unlabelled(self) -> None:
        action = Action(
            action_type=ActionType.BACK,
            rationale="r",
            target="back navigation",
            natural_language_target=None,
        )

        outcome = ActionOutcome.from_step_result(result=self.__result(action=action))

        self.assertEqual(outcome.target, "back navigation")

    def test_carries_failure_and_no_screen_change(self) -> None:
        action = Action(
            action_type=ActionType.TAP, rationale="r", natural_language_target="Dead button"
        )

        outcome = ActionOutcome.from_step_result(
            result=self.__result(action=action, success=False, screen_changed=False)
        )

        self.assertFalse(outcome.success)
        self.assertFalse(outcome.screen_changed)

    def test_carries_expected_outcome(self) -> None:
        action = Action(
            action_type=ActionType.TAP,
            rationale="r",
            natural_language_target="Checkout",
            expected_outcome=ExpectedOutcome.NEW_SCREEN,
        )

        outcome = ActionOutcome.from_step_result(result=self.__result(action=action))

        self.assertEqual(outcome.expected, ExpectedOutcome.NEW_SCREEN)

    def test_expected_defaults_to_none_when_unset(self) -> None:
        action = Action(action_type=ActionType.TAP, rationale="r", natural_language_target="Card")

        outcome = ActionOutcome.from_step_result(result=self.__result(action=action))

        self.assertIsNone(outcome.expected)

    def test_is_immutable(self) -> None:
        action = Action(action_type=ActionType.TAP, rationale="r", natural_language_target="Card")

        outcome = ActionOutcome.from_step_result(result=self.__result(action=action))

        with self.assertRaises(ValueError):
            outcome.target = "Other"  # type: ignore[misc]


class TestExplorationProgress(unittest.TestCase):
    """ExplorationProgress and TokenUsage default sanely and validate bounds."""

    def test_defaults_are_zeroed(self) -> None:
        progress = ExplorationProgress()

        self.assertEqual(progress.step, 0)
        self.assertEqual(progress.phase, BFSPhase.SCAN)
        self.assertEqual(progress.tokens.prompt, 0)
        self.assertIsNone(progress.status)

    def test_holds_a_populated_snapshot(self) -> None:
        progress = ExplorationProgress(
            step=5,
            max_steps=25,
            phase=BFSPhase.ADVANCE,
            unique_screens=12,
            coverage=48.0,
            tokens=TokenUsage(prompt=2000, completion=400, cached=100),
            status="navigating",
        )

        self.assertEqual(progress.phase, BFSPhase.ADVANCE)
        self.assertEqual(progress.tokens.completion, 400)
        self.assertEqual(progress.coverage, 48.0)

    def test_coverage_above_one_hundred_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            ExplorationProgress(coverage=150.0)


if __name__ == "__main__":
    unittest.main()
