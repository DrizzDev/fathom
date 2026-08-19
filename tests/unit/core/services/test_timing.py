from __future__ import annotations

import unittest

from fathom.constants.timing import TimingEvent, TimingPhase
from fathom.core.services.timing import RunClock
from fathom.schemas.timing import RunTimingSummary


class StepTimingTest(unittest.TestCase):
    """
    Pins the per-step commit: bucketed phases roll into one StepTiming and its dotted-key event payload.
    """

    __EXPECTED_KEYS = (
        "step.number",
        "sub_goal.index",
        "timing.ground",
        "timing.analyze",
        "timing.planner",
        "timing.vision",
        "timing.supervise",
        "timing.execute",
        "timing.observe",
        "timing.record",
        "timing.wait",
        "timing.compute",
        "timing.total",
    )

    @staticmethod
    def __clocked() -> RunClock:
        """
        Build a run clock with one fully populated pending step bucket.
        """

        clock = RunClock()
        clock.record(phase=TimingPhase.GROUND, duration=10.0)
        clock.record(phase=TimingPhase.ANALYZE, duration=20.0)
        clock.record(phase=TimingPhase.PLANNER, duration=15.0)
        clock.record(phase=TimingPhase.VISION, duration=8.0)
        clock.record(phase=TimingPhase.SUPERVISE, duration=5.0)
        clock.record(phase=TimingPhase.EXECUTE, duration=30.0)
        clock.record(phase=TimingPhase.OBSERVE, duration=12.0)
        clock.record(phase=TimingPhase.RECORD, duration=6.0)
        clock.record(phase=TimingPhase.WAIT, duration=100.0)
        return clock

    def test_step_timing_event_carries_expected_keys(self) -> None:
        """
        A committed step timing projects to exactly the required dotted-key fields under the step.timing event.
        """

        timing = self.__clocked().commit(step=3, subgoal=1)
        event = timing.to_event()

        self.assertEqual(TimingEvent.STEP.value, "step.timing")
        for key in self.__EXPECTED_KEYS:
            self.assertIn(key, event)
        self.assertEqual(event["step.number"], 3)
        self.assertEqual(event["sub_goal.index"], 1)

    def test_compute_excludes_planner_vision_and_wait(self) -> None:
        """
        Agent compute sums node phases only; planner and vision are sub-durations and the wait is carved out.
        """

        timing = self.__clocked().commit(step=0, subgoal=-1)

        self.assertEqual(timing.compute, 10.0 + 20.0 + 5.0 + 30.0 + 12.0 + 6.0)
        self.assertEqual(timing.total, timing.compute + 100.0)
        self.assertEqual(timing.planner, 15.0)
        self.assertEqual(timing.vision, 8.0)

    def test_commit_clears_pending_bucket(self) -> None:
        """
        A second commit after one populated step yields a zeroed timing.
        """

        clock = self.__clocked()
        clock.commit(step=0, subgoal=0)
        second = clock.commit(step=1, subgoal=0)

        self.assertEqual(second.compute, 0.0)
        self.assertEqual(second.total, 0.0)


class RunSummaryTest(unittest.TestCase):
    """
    Pins the run rollup: per-phase totals/means and the planner-vs-vision LLM split.
    """

    def test_summary_splits_planner_and_vision_calls(self) -> None:
        """
        The run summary counts a planner call for the analyzed step and a vision call for the observed step.
        """

        clock = RunClock()
        clock.record(phase=TimingPhase.ANALYZE, duration=20.0)
        clock.record(phase=TimingPhase.PLANNER, duration=15.0)
        clock.commit(step=0, subgoal=0)
        clock.record(phase=TimingPhase.RECORD, duration=6.0)
        clock.record(phase=TimingPhase.VISION, duration=9.0)
        clock.record(phase=TimingPhase.WAIT, duration=40.0)
        clock.commit(step=1, subgoal=0)

        summary = clock.summary()

        self.assertIsInstance(summary, RunTimingSummary)
        self.assertEqual(summary.steps, 2)
        self.assertEqual(summary.planner.calls, 1)
        self.assertEqual(summary.vision.calls, 1)
        self.assertEqual(summary.planner.duration, 15.0)
        self.assertEqual(summary.vision.duration, 9.0)
        self.assertEqual(summary.wait, 40.0)
        self.assertEqual(summary.compute, 20.0 + 6.0)
        self.assertEqual(summary.wall, summary.compute + summary.wait)
        self.assertIn(TimingPhase.ANALYZE.value, summary.phases)
        self.assertEqual(summary.phases[TimingPhase.ANALYZE.value].total, 20.0)

    def test_summary_of_empty_run_is_zeroed(self) -> None:
        """
        A run with no committed steps summarizes to zero without dividing by zero.
        """

        summary = RunClock().summary()

        self.assertEqual(summary.steps, 0)
        self.assertEqual(summary.wall, 0.0)
        self.assertEqual(summary.phases[TimingPhase.GROUND.value].mean, 0.0)


if __name__ == "__main__":
    unittest.main()
