from __future__ import annotations

import unittest

from fathom.constants.completion import GateOutcome, RetainReason
from fathom.core.agent.completion import CompletionGate
from fathom.core.services.shadow import ShadowRecorder
from fathom.schemas.completion import (
    ActionEvidence,
    ClaimEvidence,
    CompletionEvidence,
    GateDecision,
    ScreenEvidence,
)
from fathom.schemas.subgoal import SubGoal
from fathom.schemas.vision import ActionKind


class ShadowRecorderTest(unittest.TestCase):
    """
    Cover trace recording and trial mirroring through the production gate.
    """

    def setUp(self) -> None:
        """
        Build the ISSUE-004 loop-turn evidence and its live RETAIN decision.
        """

        self.sub_goal = SubGoal(description="Open the notes list", index=1)
        self.evidence = CompletionEvidence(
            claim=ClaimEvidence(asserted=False, explained=True),
            action=ActionEvidence(dispatched=True, executed=True),
            screen=ScreenEvidence(evolved=True),
        )
        self.decision = GateDecision(
            outcome=GateOutcome.RETAIN,
            retain_reason=RetainReason.MISSING_CLAIM,
        )

    def test_records_trace_without_trial(self) -> None:
        """
        Log the trace and return no comparison when no trial decider is bound.
        """

        recorder = ShadowRecorder()

        with self.assertLogs("fathom.core.services.shadow", level="INFO"):
            shadow = recorder.observe(
                workflow_id="59cd9b0b",
                turn=4,
                sub_goal=self.sub_goal,
                action_kind=ActionKind.NAVIGATION,
                evidence=self.evidence,
                decision=self.decision,
            )

        self.assertIsNone(shadow)

    def test_mirrors_agreeing_trial(self) -> None:
        """
        Report agreement when the trial gate reproduces the recorded loop-turn RETAIN.
        """

        recorder = ShadowRecorder(trial=CompletionGate())

        shadow = recorder.observe(
            workflow_id="59cd9b0b",
            turn=4,
            sub_goal=self.sub_goal,
            action_kind=ActionKind.NAVIGATION,
            evidence=self.evidence,
            decision=self.decision,
        )

        self.assertIsNotNone(shadow)
        assert shadow is not None
        self.assertTrue(shadow.agrees)

    def test_mirrors_disagreeing_trial(self) -> None:
        """
        Report disagreement when the recorded live decision departs from what the gate derives.
        """

        recorder = ShadowRecorder(trial=CompletionGate())
        complete = CompletionEvidence(
            claim=ClaimEvidence(asserted=True, explained=True),
            action=ActionEvidence(dispatched=True, executed=True),
            screen=ScreenEvidence(evolved=True),
        )

        shadow = recorder.observe(
            workflow_id="59cd9b0b",
            turn=4,
            sub_goal=self.sub_goal,
            action_kind=ActionKind.NAVIGATION,
            evidence=complete,
            decision=self.decision,
        )

        self.assertIsNotNone(shadow)
        assert shadow is not None
        self.assertFalse(shadow.agrees)
        self.assertEqual(shadow.live.outcome, GateOutcome.RETAIN)
        self.assertEqual(shadow.trial.outcome, GateOutcome.ADVANCE)
