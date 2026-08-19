"""
Monotonic per-step timing ledger for one agent run.

Nodes bracket their compute with :class:`Stopwatch` (or the :meth:`RunClock.phase` context manager),
each recording into a shared pending bucket. RECORD commits the bucket into one :class:`StepTiming`
per step; the executor emits a :class:`RunTimingSummary` at run end. Instrumentation is additive and
never influences a control decision.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Callable, Dict, Iterator, List, Tuple

from fathom.constants.timing import COMPUTE_PHASES, Scale, TimingPhase
from fathom.schemas.timing import PhaseRollup, RunTimingSummary, StepTiming, Usage


class Stopwatch:
    """
    Single-use monotonic stopwatch reporting elapsed milliseconds since construction.
    """

    def __init__(self) -> None:
        """
        Start the stopwatch at construction time.
        """

        self.__started = time.monotonic()

    @property
    def elapsed(self) -> float:
        """
        Return milliseconds elapsed since the stopwatch started.
        """

        return (time.monotonic() - self.__started) * Scale.MILLIS_PER_SECOND


class RunClock:
    """
    Accumulates per-phase durations for the active step and rolls committed steps into a run summary.
    """

    def __init__(self) -> None:
        """
        Start with an empty pending bucket and no committed step timings.
        """

        self.__steps: List[StepTiming] = []
        self.__pending: Dict[TimingPhase, float] = {}

    def record(self, *, phase: TimingPhase, duration: float) -> None:
        """
        Add a measured duration to the active step's bucket for the given phase.
        """

        self.__pending[phase] = self.__pending.get(phase, 0.0) + max(0.0, duration)

    @contextmanager
    def phase(self, phase: TimingPhase) -> Iterator[None]:
        """
        Time the wrapped block and record it against the phase on exit, however the block returns.
        """

        watch = Stopwatch()

        try:
            yield
        finally:
            self.record(phase=phase, duration=watch.elapsed)

    def commit(self, *, step: int, subgoal: int) -> StepTiming:
        """
        Snapshot the pending bucket into one step timing, clear it, and retain it for the run summary.
        """

        compute = sum(self.__pending.get(phase, 0.0) for phase in COMPUTE_PHASES)

        timing = StepTiming(
            step=step,
            subgoal=subgoal,
            compute=compute,
            wait=self.__take(phase=TimingPhase.WAIT),
            ground=self.__take(phase=TimingPhase.GROUND),
            vision=self.__take(phase=TimingPhase.VISION),
            record=self.__take(phase=TimingPhase.RECORD),
            analyze=self.__take(phase=TimingPhase.ANALYZE),
            planner=self.__take(phase=TimingPhase.PLANNER),
            execute=self.__take(phase=TimingPhase.EXECUTE),
            observe=self.__take(phase=TimingPhase.OBSERVE),
            supervise=self.__take(phase=TimingPhase.SUPERVISE),
            total=compute + self.__pending.get(TimingPhase.WAIT, 0.0),
        )
        self.__pending = {}
        self.__steps.append(timing)

        return timing

    def summary(self) -> RunTimingSummary:
        """
        Roll every committed step into per-phase totals, means, shares, and planner/vision LLM splits.
        """

        steps = len(self.__steps)
        wait = sum(step.wait for step in self.__steps)
        compute = sum(step.compute for step in self.__steps)

        return RunTimingSummary(
            wait=wait,
            steps=steps,
            compute=compute,
            wall=compute + wait,
            vision=self.__usage(reader=lambda step: step.vision),
            planner=self.__usage(reader=lambda step: step.planner),
            phases=self.__phase_rollups(steps=steps, compute=compute),
        )

    @property
    def steps(self) -> Tuple[StepTiming, ...]:
        """
        Return the committed step timings in commit order.
        """

        return tuple(self.__steps)

    def __usage(self, *, reader: Callable[[StepTiming], float]) -> Usage:
        """
        Fold one LLM sub-duration across steps into a call count and total duration.
        """

        durations = [reader(step) for step in self.__steps]
        return Usage(
            duration=sum(durations),
            calls=sum(1 for duration in durations if duration > 0.0),
        )

    def __phase_rollups(self, *, steps: int, compute: float) -> Dict[str, PhaseRollup]:
        """
        Build the per-phase total/mean/share rollup keyed by phase name.
        """

        rollups: Dict[str, PhaseRollup] = {}

        for phase in TimingPhase:
            total = sum(self.__phase_value(step=step, phase=phase) for step in self.__steps)
            rollups[phase.value] = PhaseRollup(
                total=total,
                mean=total / steps if steps else 0.0,
                share=(total / compute * Scale.PERCENT) if compute else 0.0,
            )

        return rollups

    @staticmethod
    def __phase_value(*, step: StepTiming, phase: TimingPhase) -> float:
        """
        Read one phase's duration off a committed step timing.
        """

        return float(getattr(step, phase.value))

    def __take(self, *, phase: TimingPhase) -> float:
        """
        Read the pending duration for a phase without clearing the bucket.
        """

        return self.__pending.get(phase, 0.0)
