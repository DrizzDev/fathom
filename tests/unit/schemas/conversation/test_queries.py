from __future__ import annotations

import unittest

from pydantic import ValidationError

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
                title="t" * (THREAD_TITLE_PREFIX_MAX_LENGTH + 1),
            )
