from __future__ import annotations

import unittest

from fathom.constants.exploration import ExpectedOutcome, FocusRelevance


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


class TestFocusRelevance(unittest.TestCase):
    """FocusRelevance ranks frontier screens for focus-aware recovery."""

    def test_recovery_priority_orders_on_focus_before_off_focus(self) -> None:
        ordered = sorted(FocusRelevance, key=lambda relevance: relevance.recovery_priority)

        self.assertEqual(
            ordered,
            [
                FocusRelevance.ON_FOCUS,
                FocusRelevance.LEADS_TOWARD,
                FocusRelevance.UNSCOPED,
                FocusRelevance.OFF_FOCUS,
            ],
        )

    def test_off_focus_outranks_every_other_tier(self) -> None:
        off = FocusRelevance.OFF_FOCUS.recovery_priority
        for relevance in (
            FocusRelevance.ON_FOCUS,
            FocusRelevance.LEADS_TOWARD,
            FocusRelevance.UNSCOPED,
        ):
            self.assertLess(relevance.recovery_priority, off, relevance)


if __name__ == "__main__":
    unittest.main()
