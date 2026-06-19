from __future__ import annotations

import unittest
from typing import List

from fathom.core.runtime.emitter import RuntimeEventEmitter
from fathom.interfaces.journal import RuntimeJournalPort
from fathom.schemas.events import RuntimeEvent, RuntimeEventKind


class _RecordingJournal(RuntimeJournalPort):
    """
    Minimal RuntimeJournalPort that records every event passed to it.
    """

    def __init__(self) -> None:
        """
        Initialize the recording journal with an empty buffer.
        """

        self.records: List[RuntimeEvent] = []

    async def record(self, *, event: RuntimeEvent) -> None:
        """
        Append the event to the recording buffer.
        """

        self.records.append(event)


class _FailingJournal(RuntimeJournalPort):
    """
    RuntimeJournalPort that raises a deterministic exception on every record.
    """

    async def record(self, *, event: RuntimeEvent) -> None:
        """
        Raise a RuntimeError to exercise the emitter's exception isolation.
        """

        _ = event
        raise RuntimeError("journal down")


class RuntimeEventEmitterTest(unittest.IsolatedAsyncioTestCase):
    """
    Pins for the RuntimeEventEmitter event-construction and journal-isolation contract.
    """

    async def test_emit_records_event_through_journal(self) -> None:
        """
        emit() must dispatch a typed RuntimeEvent to the journal port.
        """

        journal = _RecordingJournal()
        emitter = RuntimeEventEmitter(workflow_id="run-1", journal=journal)

        await emitter.emit(
            step=4,
            payload={"reason": "test"},
            kind=RuntimeEventKind.DECISION_MADE,
            correlation_id="cid-1",
        )

        self.assertEqual(len(journal.records), 1)
        event = journal.records[0]
        self.assertEqual(event.step, 4)
        self.assertEqual(event.kind, RuntimeEventKind.DECISION_MADE)
        self.assertEqual(event.workflow, "run-1")

    async def test_emit_uses_correlation_id_verbatim(self) -> None:
        """
        Supplied correlation_id values become the event identifier verbatim.
        """

        journal = _RecordingJournal()
        emitter = RuntimeEventEmitter(workflow_id="run-1", journal=journal)

        await emitter.emit(
            step=0,
            payload=None,
            kind=RuntimeEventKind.SCREEN_OBSERVED,
            correlation_id="stable-id",
        )

        self.assertEqual(journal.records[0].identifier, "stable-id")

    async def test_emit_generates_uuid_when_correlation_id_missing(self) -> None:
        """
        Omitted correlation_id values trigger a fresh non-empty identifier.
        """

        journal = _RecordingJournal()
        emitter = RuntimeEventEmitter(workflow_id="run-1", journal=journal)

        await emitter.emit(
            step=0,
            payload=None,
            kind=RuntimeEventKind.SCREEN_OBSERVED,
        )
        await emitter.emit(
            step=0,
            payload=None,
            kind=RuntimeEventKind.SCREEN_OBSERVED,
        )

        first_id = journal.records[0].identifier
        second_id = journal.records[1].identifier
        self.assertTrue(first_id)
        self.assertTrue(second_id)
        self.assertNotEqual(first_id, second_id)

    async def test_emit_swallows_journal_exception(self) -> None:
        """
        emit() must not propagate journal exceptions to the caller.
        """

        emitter = RuntimeEventEmitter(workflow_id="run-1", journal=_FailingJournal())

        await emitter.emit(
            step=1,
            payload={"reason": "fail"},
            kind=RuntimeEventKind.DECISION_MADE,
            correlation_id="cid",
        )
