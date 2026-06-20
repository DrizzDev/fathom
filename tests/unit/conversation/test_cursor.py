from __future__ import annotations

import unittest

from fathom.conversation.cursor import CompositeTimelineCursor, OpaqueCursor
from fathom.core.exceptions import InteractionError


class TestCompositeTimelineCursor(unittest.TestCase):
    """
    Unit tests for the per-kind composite timeline cursor.
    """

    def test_round_trip_preserves_per_kind_positions(self) -> None:
        """
        Round-trip an encoded composite cursor through decode and back.
        """

        composite = CompositeTimelineCursor(
            messages="m-cursor",
            events="e-cursor",
            artifacts="a-cursor",
            contexts="c-cursor",
        )

        decoded = CompositeTimelineCursor.decode(value=composite.encode())

        self.assertEqual("m-cursor", decoded.messages)
        self.assertEqual("e-cursor", decoded.events)
        self.assertEqual("a-cursor", decoded.artifacts)
        self.assertEqual("c-cursor", decoded.contexts)

    def test_empty_composite_reports_empty(self) -> None:
        """
        Identify an empty composite when no per-kind position is set.
        """

        composite = CompositeTimelineCursor()

        self.assertTrue(composite.is_empty())

    def test_partially_populated_composite_is_not_empty(self) -> None:
        """
        Detect a non-empty composite when at least one position is set.
        """

        composite = CompositeTimelineCursor(messages="m-cursor")

        self.assertFalse(composite.is_empty())
        self.assertIsNone(composite.events)
        self.assertIsNone(composite.artifacts)
        self.assertIsNone(composite.contexts)

    def test_decode_rejects_invalid_token(self) -> None:
        """
        Reject malformed cursor tokens with a typed interaction error.
        """

        with self.assertRaises(InteractionError):
            CompositeTimelineCursor.decode(value="not-a-real-cursor")

    def test_decode_rejects_non_string_sub_cursor(self) -> None:
        """
        Reject decoded payloads where a sub-cursor is not a string.
        """

        import base64
        import json

        from fathom.constants.conversation import CURSOR_VERSION

        bad = base64.urlsafe_b64encode(
            json.dumps({"v": CURSOR_VERSION, "messages": 5}).encode()
        ).decode("ascii")

        with self.assertRaises(InteractionError):
            CompositeTimelineCursor.decode(value=bad)

    def test_sub_cursor_compatible_with_opaque_cursor(self) -> None:
        """
        Sub-cursor strings remain compatible with OpaqueCursor encoding.
        """

        from datetime import datetime, timezone

        opaque = OpaqueCursor(
            created=datetime(2026, 5, 4, tzinfo=timezone.utc),
            identifier="entry-1",
        ).encode()
        composite = CompositeTimelineCursor(messages=opaque)

        decoded = CompositeTimelineCursor.decode(value=composite.encode())
        rehydrated = OpaqueCursor.decode(value=decoded.messages or "")

        self.assertEqual("entry-1", rehydrated.identifier)
