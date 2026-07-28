from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from fathom.adapters.replay.corpus import Corpus
from fathom.constants.completion import GateOutcome, RetainReason
from fathom.schemas.completion import (
    ActionEvidence,
    ClaimEvidence,
    CompletionEvidence,
    ScreenEvidence,
)
from fathom.schemas.shadow import Reading, Tape, Trace
from fathom.schemas.subgoal import SubGoal, SubGoalKind
from fathom.schemas.vision import ActionKind


class CorpusLoadTest(unittest.TestCase):
    """
    Cover typed tape loading from a corpus directory.
    """

    def setUp(self) -> None:
        """
        Create a temporary corpus root.
        """

        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def test_loads_typed_tapes_in_name_order(self) -> None:
        """
        Restore tapes from typed files while ignoring non-tape files.
        """

        tape = Tape(
            run="59cd9b0b",
            traces=[
                Trace(
                    turn=0,
                    task=SubGoal(description="Open the notes list", index=1),
                    kind=ActionKind.NAVIGATION,
                    evidence=CompletionEvidence(
                        claim=ClaimEvidence(asserted=False, explained=True),
                        action=ActionEvidence(dispatched=True, executed=True),
                        screen=ScreenEvidence(evolved=True),
                    ),
                    reading=Reading(
                        outcome=GateOutcome.RETAIN,
                        reason=RetainReason.MISSING_CLAIM,
                    ),
                )
            ],
        )
        (self.root / "59cd9b0b.tape.json").write_text(tape.model_dump_json())
        (self.root / "notes.txt").write_text("not a tape")

        tapes = Corpus(root=self.root).load()

        self.assertEqual(tapes, [tape])


class CorpusLegacyTest(unittest.TestCase):
    """
    Cover conversion of the POC extraction archive into typed tapes.
    """

    def setUp(self) -> None:
        """
        Write a representative extraction archive to disk.
        """

        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "tapes.json"
        self.path.write_text(
            json.dumps(
                {
                    "59cd9b0b": [
                        {
                            "ts": "13:11:17",
                            "sg": 0,
                            "kind": "action",
                            "outcome": "ADVANCE",
                            "reason": "STRICT_PATH",
                            "claim": True,
                            "explained": True,
                            "dispatched": True,
                            "evolved": True,
                            "changed": True,
                            "effect": "progress",
                            "act": "navigation",
                            "veto": False,
                        },
                        {
                            "ts": "13:12:02",
                            "sg": 1,
                            "kind": "action",
                            "outcome": "RETAIN",
                            "reason": "MISSING_CLAIM",
                            "claim": False,
                            "explained": True,
                            "dispatched": True,
                            "evolved": True,
                            "changed": True,
                            "effect": "progress",
                            "act": "mystery",
                            "veto": False,
                        },
                        {
                            "ts": "13:12:40",
                            "sg": 2,
                            "kind": "validation",
                            "outcome": "RETAIN",
                            "reason": "MISSING_VALIDATION",
                            "claim": False,
                            "explained": False,
                            "dispatched": False,
                            "evolved": False,
                            "changed": False,
                            "effect": "no_progress",
                            "act": "validation",
                            "veto": False,
                        },
                        {
                            "ts": "13:13:05",
                            "sg": 2,
                            "kind": "action",
                            "outcome": "UNPARSED",
                        },
                    ]
                }
            )
        )

    def test_converts_turns_and_drops_evidence_free_rows(self) -> None:
        """
        Map booleans, enums, and readings while dropping rows without evidence.
        """

        tapes = Corpus.legacy(path=self.path)

        self.assertEqual(len(tapes), 1)
        tape = tapes[0]
        self.assertEqual(tape.run, "59cd9b0b")
        self.assertEqual(len(tape.traces), 3)

        advance = tape.traces[0]
        self.assertEqual(advance.kind, ActionKind.NAVIGATION)
        self.assertEqual(advance.reading, Reading(outcome=GateOutcome.ADVANCE, reason=None))
        self.assertTrue(advance.evidence.claim.asserted)

        retained = tape.traces[1]
        self.assertEqual(retained.kind, ActionKind.UNKNOWN)
        self.assertEqual(retained.task.index, 1)
        self.assertEqual(retained.task.description, "")
        self.assertFalse(retained.evidence.claim.asserted)
        self.assertEqual(
            retained.reading,
            Reading(outcome=GateOutcome.RETAIN, reason=RetainReason.MISSING_CLAIM),
        )

        validation = tape.traces[2]
        self.assertEqual(validation.task.kind, SubGoalKind.VALIDATION)
        self.assertFalse(validation.evidence.validation.executed)
        self.assertEqual(
            validation.reading,
            Reading(outcome=GateOutcome.RETAIN, reason=RetainReason.MISSING_VALIDATION),
        )
