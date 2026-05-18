from __future__ import annotations

import unittest

from fathom.core.runtime.healing import HealingUsage


class HealingUsageTest(unittest.TestCase):
    """
    Pins for the runtime HealingUsage per-task and per-run accounting.
    """

    def test_record_increments_per_task_and_per_run(self) -> None:
        """
        record() must increment both the per-task and per-run counters.
        """

        usage = HealingUsage()
        usage.record(task_id="task-a")
        usage.record(task_id="task-a")
        usage.record(task_id="task-b")

        self.assertEqual(usage.task_count(task_id="task-a"), 2)
        self.assertEqual(usage.task_count(task_id="task-b"), 1)
        self.assertEqual(usage.run_count(), 3)

    def test_task_count_returns_zero_for_unknown_task(self) -> None:
        """
        task_count() for a task that never recorded must return zero.
        """

        self.assertEqual(HealingUsage().task_count(task_id="missing"), 0)

    def test_reset_task_clears_only_that_task(self) -> None:
        """
        reset_task() clears one task's counter without affecting other tasks or the run total.
        """

        usage = HealingUsage()
        usage.record(task_id="task-a")
        usage.record(task_id="task-b")
        usage.reset_task(task_id="task-a")

        self.assertEqual(usage.task_count(task_id="task-a"), 0)
        self.assertEqual(usage.task_count(task_id="task-b"), 1)
        self.assertEqual(usage.run_count(), 2)

    def test_to_state_round_trip(self) -> None:
        """
        load_state() must restore the counters that to_state() serialized.
        """

        original = HealingUsage()
        original.record(task_id="task-a")
        original.record(task_id="task-b")
        original.record(task_id="task-b")

        restored = HealingUsage()
        restored.load_state(state=original.to_state())

        self.assertEqual(restored.task_count(task_id="task-a"), 1)
        self.assertEqual(restored.task_count(task_id="task-b"), 2)
        self.assertEqual(restored.run_count(), 3)

    def test_load_state_rejects_missing_per_run(self) -> None:
        """
        load_state() must raise ValueError when the payload omits 'per_run'.
        """

        with self.assertRaises(ValueError):
            HealingUsage().load_state(state={"per_task": {}})

    def test_load_state_rejects_missing_per_task(self) -> None:
        """
        load_state() must raise ValueError when the payload omits 'per_task'.
        """

        with self.assertRaises(ValueError):
            HealingUsage().load_state(state={"per_run": 0})
