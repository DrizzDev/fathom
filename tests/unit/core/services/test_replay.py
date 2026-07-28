from __future__ import annotations

import unittest

from fathom.constants.completion import GateOutcome, RetainReason
from fathom.core.agent.completion import CompletionGate
from fathom.core.services.replay import Replayer
from fathom.schemas.completion import (
    ActionEvidence,
    ClaimEvidence,
    CompletionEvidence,
    ScreenEvidence,
    ValidationEvidence,
)
from fathom.schemas.shadow import Reading, Tape, Trace
from fathom.schemas.subgoal import SubGoal, SubGoalKind
from fathom.schemas.vision import ActionKind


class ReplayerTest(unittest.TestCase):
    """
    Cover tape replay through the live completion gate.
    """

    def setUp(self) -> None:
        """
        Build the replayer around the production gate.
        """

        self.replayer = Replayer(decider=CompletionGate())

    def test_matches_recorded_missing_claim_loop_turn(self) -> None:
        """
        Reproduce the ISSUE-004 loop turn: work done, boolean forgotten, RETAIN(MISSING_CLAIM).
        """

        tape = Tape(
            run="59cd9b0b",
            traces=[
                self.__trace(
                    turn=0,
                    asserted=False,
                    explained=True,
                    dispatched=True,
                    evolved=True,
                    reading=Reading(
                        outcome=GateOutcome.RETAIN,
                        reason=RetainReason.MISSING_CLAIM,
                    ),
                )
            ],
        )

        parity = self.replayer.replay(tape=tape)

        self.assertEqual(parity.total, 1)
        self.assertEqual(parity.matched, 1)
        self.assertEqual(parity.divergences, [])

    def test_matches_recorded_strict_path_advance(self) -> None:
        """
        Reproduce a clean advance: claim asserted, explained, and screen-verified dispatch.
        """

        tape = Tape(
            run="59cd9b0b",
            traces=[
                self.__trace(
                    turn=1,
                    asserted=True,
                    explained=True,
                    dispatched=True,
                    evolved=True,
                    reading=Reading(outcome=GateOutcome.ADVANCE, reason=None),
                )
            ],
        )

        parity = self.replayer.replay(tape=tape)

        self.assertEqual(parity.matched, 1)

    def test_matches_recorded_durable_outcome_retention(self) -> None:
        """
        Reproduce the durable path: an effective save still retains awaiting outcome evidence.
        """

        trace = Trace(
            turn=2,
            task=SubGoal(description="Save the note", index=2),
            kind=ActionKind.INPUT,
            evidence=self.__evidence(
                asserted=True,
                explained=True,
                dispatched=True,
                evolved=True,
            ),
            reading=Reading(
                outcome=GateOutcome.RETAIN,
                reason=RetainReason.MISSING_OUTCOME_EVIDENCE,
            ),
        )

        parity = self.replayer.replay(tape=Tape(run="154ec8a1", traces=[trace]))

        self.assertEqual(parity.matched, 1)

    def test_matches_recorded_validation_advance(self) -> None:
        """
        Reproduce a validation sub-goal advancing on an executed validate command.
        """

        trace = Trace(
            turn=3,
            task=SubGoal(
                description="Verify the note exists", index=3, kind=SubGoalKind.VALIDATION
            ),
            kind=ActionKind.VALIDATION,
            evidence=CompletionEvidence(
                claim=ClaimEvidence(asserted=True, explained=True),
                action=ActionEvidence(dispatched=True, executed=True),
                screen=ScreenEvidence(evolved=False),
                validation=ValidationEvidence(executed=True),
            ),
            reading=Reading(outcome=GateOutcome.ADVANCE, reason=None),
        )

        parity = self.replayer.replay(tape=Tape(run="154ec8a1", traces=[trace]))

        self.assertEqual(parity.matched, 1)

    def test_reports_divergence_against_tampered_recording(self) -> None:
        """
        Surface a divergence row carrying turn, live, and trial readings.
        """

        tape = Tape(
            run="6033fea1",
            traces=[
                self.__trace(
                    turn=5,
                    asserted=False,
                    explained=True,
                    dispatched=True,
                    evolved=True,
                    reading=Reading(outcome=GateOutcome.ADVANCE, reason=None),
                )
            ],
        )

        parity = self.replayer.replay(tape=tape)

        self.assertEqual(parity.matched, 0)
        self.assertEqual(len(parity.divergences), 1)
        divergence = parity.divergences[0]
        self.assertEqual(divergence.turn, 5)
        self.assertEqual(divergence.live.outcome, GateOutcome.ADVANCE)
        self.assertEqual(divergence.trial.outcome, GateOutcome.RETAIN)
        self.assertEqual(divergence.trial.reason, RetainReason.MISSING_CLAIM)

    @classmethod
    def __trace(
        cls,
        *,
        turn: int,
        asserted: bool,
        explained: bool,
        dispatched: bool,
        evolved: bool,
        reading: Reading,
    ) -> Trace:
        """
        Build an action-sub-goal trace with a transient navigation description.
        """

        return Trace(
            turn=turn,
            task=SubGoal(description="Open the notes list", index=1),
            kind=ActionKind.NAVIGATION,
            evidence=cls.__evidence(
                asserted=asserted,
                explained=explained,
                dispatched=dispatched,
                evolved=evolved,
            ),
            reading=reading,
        )

    @staticmethod
    def __evidence(
        *,
        asserted: bool,
        explained: bool,
        dispatched: bool,
        evolved: bool,
    ) -> CompletionEvidence:
        """
        Build an evidence bundle from the four gate-visible signals.
        """

        return CompletionEvidence(
            claim=ClaimEvidence(asserted=asserted, explained=explained),
            action=ActionEvidence(dispatched=dispatched, executed=dispatched),
            screen=ScreenEvidence(evolved=evolved),
        )
