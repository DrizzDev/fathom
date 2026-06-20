from __future__ import annotations

import json
import unittest
from datetime import datetime, timezone
from typing import Any, Dict

from fathom.core.exceptions import InteractionError
from fathom.infrastructure.interaction.pypika.sqlite.row import RowMapper


class _FakeRow:
    """
    Minimal mapping-style row used to drive the row mapper without sqlite.
    """

    def __init__(self, *, fields: Dict[str, Any]) -> None:
        """
        Bind the field values exposed via subscript access.
        """

        self.__fields = fields

    def __getitem__(self, key: str) -> Any:
        """
        Return the stored value for the requested column key.
        """

        return self.__fields[key]


class TestRowMapperEnumGuard(unittest.TestCase):
    """
    Row mapper must raise InteractionError when a stored enum value is invalid.
    """

    def setUp(self) -> None:
        """
        Build the mapper under test and a deterministic timestamp.
        """

        self.__mapper = RowMapper()
        self.__now = datetime(2026, 6, 8, tzinfo=timezone.utc).isoformat()

    def __thread_row(self, *, state: str) -> _FakeRow:
        """
        Build a thread row with the supplied state value.
        """

        return _FakeRow(
            fields={
                "id": "thread-1",
                "tenant": "tenant-1",
                "workspace": None,
                "title": "demo",
                "state": state,
                "digest": None,
                "cursor": 0,
                "creator": None,
                "created_at": self.__now,
                "updated_at": self.__now,
                "archived_at": None,
                "deleted_at": None,
                "metadata": json.dumps({}),
            }
        )

    def test_valid_state_returns_thread(self) -> None:
        """
        A known state value must round-trip through the mapper unchanged.
        """

        thread = self.__mapper.thread(row=self.__thread_row(state="active"))

        self.assertEqual("thread-1", thread.identity.id)
        self.assertEqual("active", thread.state.value)

    def test_unknown_state_raises_interaction_error(self) -> None:
        """
        An unknown state value must raise InteractionError with the offending field.
        """

        with self.assertRaises(InteractionError) as guard:
            self.__mapper.thread(row=self.__thread_row(state="not-a-state"))

        self.assertIn("state", str(guard.exception))
        self.assertIn("ThreadState", str(guard.exception))
