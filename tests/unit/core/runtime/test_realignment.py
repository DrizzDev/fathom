from __future__ import annotations

import unittest

from fathom.core.runtime.realignment import RealignmentTracker


class RealignmentTrackerTest(unittest.TestCase):
    """
    Pins for the runtime RealignmentTracker budget semantics.
    """

    def test_initial_state_count_zero_and_not_exhausted(self) -> None:
        """
        A fresh tracker reports zero count and is not exhausted.
        """

        tracker = RealignmentTracker(budget=3)

        self.assertEqual(tracker.count, 0)
        self.assertEqual(tracker.budget, 3)
        self.assertFalse(tracker.exhausted())

    def test_record_increments_count(self) -> None:
        """
        record() increments the count by one.
        """

        tracker = RealignmentTracker(budget=3)
        tracker.record()
        tracker.record()

        self.assertEqual(tracker.count, 2)
        self.assertFalse(tracker.exhausted())

    def test_exhausted_when_count_reaches_budget(self) -> None:
        """
        exhausted() turns true once count >= budget.
        """

        tracker = RealignmentTracker(budget=2)
        tracker.record()
        tracker.record()

        self.assertTrue(tracker.exhausted())

    def test_to_state_round_trip(self) -> None:
        """
        load_state() must restore the budget and count from to_state().
        """

        original = RealignmentTracker(budget=5)
        original.record()
        original.record()

        restored = RealignmentTracker()
        restored.load_state(state=original.to_state())

        self.assertEqual(restored.budget, 5)
        self.assertEqual(restored.count, 2)
        self.assertFalse(restored.exhausted())
