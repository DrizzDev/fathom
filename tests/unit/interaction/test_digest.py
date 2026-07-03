from __future__ import annotations

import unittest
from datetime import datetime, timezone

from fathom.constants.collaboration import EventKind, EventSource
from fathom.interaction.digest import EventDigest
from fathom.schemas.interaction import Metadata


class TestEventDigest(unittest.TestCase):
    """
    EventDigest produces a canonical, deterministic digest across backends.
    """

    def setUp(self) -> None:
        """
        Bind a deterministic timestamp and the collaborator under test.
        """

        self.__digest = EventDigest()
        self.__now = datetime(2026, 6, 8, 9, 30, tzinfo=timezone.utc)

    def __compute(self, *, payload: Metadata, sequence: int = 1) -> str:
        """
        Compute a digest with the standard test fixture inputs.
        """

        return self.__digest.compute(
            kind=EventKind.THREAD_CREATED,
            source=EventSource.FATHOM,
            payload=payload,
            created=self.__now,
            sequence=sequence,
        )

    def test_same_inputs_produce_identical_digest(self) -> None:
        """
        Identical event inputs must produce byte-identical digests.
        """

        first = self.__compute(payload=Metadata(entries={"actor": "actor-1"}))
        second = self.__compute(payload=Metadata(entries={"actor": "actor-1"}))

        self.assertEqual(first, second)

    def test_payload_key_order_does_not_affect_digest(self) -> None:
        """
        Two payloads with the same keys in different insert order must hash the same.
        """

        ordered = self.__compute(payload=Metadata(entries={"a": 1, "b": 2}))
        reordered = self.__compute(payload=Metadata(entries={"b": 2, "a": 1}))

        self.assertEqual(ordered, reordered)

    def test_equivalent_offset_timestamps_produce_identical_digest(self) -> None:
        """
        Two datetimes pointing at the same instant in different offsets must hash equally.
        """

        from datetime import timedelta

        utc = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
        ist = utc.astimezone(timezone(timedelta(hours=5, minutes=30)))

        utc_digest = self.__digest.compute(
            kind=EventKind.THREAD_CREATED,
            source=EventSource.FATHOM,
            payload=Metadata(entries={}),
            created=utc,
            sequence=1,
        )
        ist_digest = self.__digest.compute(
            kind=EventKind.THREAD_CREATED,
            source=EventSource.FATHOM,
            payload=Metadata(entries={}),
            created=ist,
            sequence=1,
        )

        self.assertEqual(utc_digest, ist_digest)

    def test_naive_timestamp_rejected(self) -> None:
        """
        Naive datetimes must be rejected so callers do not stamp ambiguous wall time.
        """

        from fathom.core.exceptions import InteractionError

        with self.assertRaises(InteractionError):
            self.__digest.compute(
                kind=EventKind.THREAD_CREATED,
                source=EventSource.FATHOM,
                payload=Metadata(entries={}),
                created=datetime(2026, 1, 1, 12, 0),
                sequence=1,
            )

    def test_sequence_change_yields_different_digest(self) -> None:
        """
        Different sequences for otherwise-identical events must hash differently.
        """

        first = self.__compute(payload=Metadata(entries={}), sequence=1)
        second = self.__compute(payload=Metadata(entries={}), sequence=2)

        self.assertNotEqual(first, second)
