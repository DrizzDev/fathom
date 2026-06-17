from __future__ import annotations

import unittest

from fathom.constants.exploration import ExpectedOutcome


class TestExpectedOutcome(unittest.TestCase):
    """ExpectedOutcome classifies which predictions imply a visible transition."""

    def test_navigational_predictions_imply_transition(self) -> None:
        for outcome in (
            ExpectedOutcome.NEW_SCREEN,
            ExpectedOutcome.DIALOG_OR_POPUP,
            ExpectedOutcome.DISMISS_OVERLAY,
            ExpectedOutcome.GO_BACK,
        ):
            self.assertTrue(outcome.implies_transition, outcome)

    def test_in_place_predictions_do_not_imply_transition(self) -> None:
        for outcome in (ExpectedOutcome.IN_SCREEN_CHANGE, ExpectedOutcome.SCROLL_CONTENT):
            self.assertFalse(outcome.implies_transition, outcome)


if __name__ == "__main__":
    unittest.main()
