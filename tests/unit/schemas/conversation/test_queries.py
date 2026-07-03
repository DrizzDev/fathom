from __future__ import annotations

import unittest

from pydantic import ValidationError

from fathom.constants.collaboration import ThreadState
from fathom.constants.conversation import THREAD_TITLE_PREFIX_MAX_LENGTH
from fathom.schemas.conversation import ConversationListQuery


class TestConversationListQueryTitleBound(unittest.TestCase):
    """
    The title filter has a length bound enforced at the boundary.
    """

    def test_title_at_max_length_is_accepted(self) -> None:
        """
        Exactly MAX_LENGTH characters must pass validation.
        """

        query = ConversationListQuery(
            tenant="tenant-1",
            operator="actor-1",
            title="t" * THREAD_TITLE_PREFIX_MAX_LENGTH,
        )

        self.assertEqual(THREAD_TITLE_PREFIX_MAX_LENGTH, len(query.title or ""))

    def test_title_over_max_length_is_rejected(self) -> None:
        """
        One character beyond MAX_LENGTH must raise ValidationError.
        """

        with self.assertRaises(ValidationError):
            ConversationListQuery(
                tenant="tenant-1",
                operator="actor-1",
                title="t" * (THREAD_TITLE_PREFIX_MAX_LENGTH + 1),
            )

    def test_state_accepts_thread_state_enum(self) -> None:
        """
        Thread lifecycle filters must be typed at the boundary.
        """

        query = ConversationListQuery(
            tenant="tenant-1",
            operator="actor-1",
            state=ThreadState.ARCHIVED,
        )

        self.assertEqual(ThreadState.ARCHIVED, query.state)

    def test_unknown_state_is_rejected(self) -> None:
        """
        Unknown thread lifecycle filters must fail before application logic.
        """

        with self.assertRaises(ValidationError):
            ConversationListQuery(
                tenant="tenant-1",
                operator="actor-1",
                state="unknown",
            )
