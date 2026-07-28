from __future__ import annotations

import unittest
from pathlib import Path
from typing import Dict, List, Tuple

from fathom.adapters.replay.corpus import Corpus
from fathom.constants.completion import GateOutcome
from fathom.constants.turn.advancement import AdvanceThreshold
from fathom.core.agent.advancement import AdvancementTrial
from fathom.core.capability.catalog import CommandCatalogProvider
from fathom.schemas.shadow import Tape, Trace

# (run, worst sub-goal index, exact recorded consecutive-RETAIN loop length) — measured from the tapes.
DocumentedLoop = Tuple[str, int, int]
DOCUMENTED_LOOPS: Tuple[DocumentedLoop, ...] = (
    ("59cd9b0b", 1, 6),
    ("6033fea1", 2, 15),
    ("44abb3b9", 3, 4),
    ("9c8704be", 1, 4),
    ("154ec8a1", 1, 5),
)


class AdvancementIssueReproductionTest(unittest.TestCase):
    """
    Reproduce the documented production loops from the recorded runs and prove the new policy bounds them.

    Each run had a sub-goal that never advanced for many consecutive turns while the live gate kept
    RETAINing it. This replays that exact recorded turn sequence through the streak-tracking trial
    adapter and asserts the loop now terminates within the backstop limit.
    """

    __TAPES = Path("debug/poc_advancement/tapes.json")

    def setUp(self) -> None:
        """
        Load the recorded corpus, or skip when the debug tapes are absent.
        """

        if not self.__TAPES.exists():
            self.skipTest("recorded advancement tapes absent (debug fixtures are gitignored).")

        self.catalog = CommandCatalogProvider().build()
        self.tapes = {tape.run: tape for tape in Corpus.legacy(path=self.__TAPES)}

    def test_recorded_loops_are_reproduced_then_bounded(self) -> None:
        """
        For every documented run: confirm the exact recorded loop, then prove the new policy caps it.
        """

        limit = int(AdvanceThreshold.RETAIN_ESCALATION)

        for run, index, loop_length in DOCUMENTED_LOOPS:
            with self.subTest(run=run):
                traces = self.__sub_goal_traces(tape=self.tapes[run], index=index)

                # 1. Reproduce the real failure: the live gate looped this many turns in a row.
                recorded = [trace.reading.outcome for trace in traces]
                self.assertEqual(
                    self.__longest_retain_run(outcomes=recorded),
                    loop_length,
                    msg=f"{run}: recorded loop length changed from the documented value",
                )

                # 2. Prove the fix: the new policy escalates and never retains past the backstop limit.
                decided = self.__replay(traces=traces)
                self.assertIn(
                    GateOutcome.FAIL,
                    decided,
                    msg=f"{run}: new policy never escalated; the loop is still unbounded",
                )
                self.assertLessEqual(
                    self.__longest_retain_run(outcomes=decided),
                    limit,
                    msg=f"{run}: new policy retained past the backstop limit before escalating",
                )

    def test_worst_sub_goal_matches_the_documented_index(self) -> None:
        """
        Guard the fixtures: the worst-looping sub-goal per run stays the one the assertions target.
        """

        for run, index, _ in DOCUMENTED_LOOPS:
            with self.subTest(run=run):
                self.assertEqual(self.__worst_index(tape=self.tapes[run]), index)

    def __replay(self, *, traces: List[Trace]) -> List[GateOutcome]:
        """
        Replay one sub-goal's recorded turns through a fresh streak-tracking trial, in order.
        """

        trial = AdvancementTrial(catalog=self.catalog)
        return [
            trial.adjudicate(
                sub_goal=trace.task,
                action_kind=trace.kind,
                evidence=trace.evidence,
            ).outcome
            for trace in traces
        ]

    @classmethod
    def __worst_index(cls, *, tape: Tape) -> int:
        """
        Return the sub-goal index with the longest consecutive-RETAIN run in the recording.
        """

        by_index: Dict[int, List[GateOutcome]] = {}
        for trace in tape.traces:
            by_index.setdefault(trace.task.index, []).append(trace.reading.outcome)

        return max(by_index, key=lambda index: cls.__longest_retain_run(outcomes=by_index[index]))

    @staticmethod
    def __longest_retain_run(*, outcomes: List[GateOutcome]) -> int:
        """
        Return the longest run of consecutive RETAIN outcomes; a sub-goal that never advances loops.
        """

        streak = 0
        longest = 0
        for outcome in outcomes:
            if outcome is GateOutcome.RETAIN:
                streak += 1
                longest = max(longest, streak)
            else:
                streak = 0

        return longest

    @staticmethod
    def __sub_goal_traces(*, tape: Tape, index: int) -> List[Trace]:
        """
        Return the recorded traces for one sub-goal, in turn order.
        """

        return [trace for trace in tape.traces if trace.task.index == index]
