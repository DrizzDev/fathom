from __future__ import annotations

import unittest
from datetime import datetime, timezone

from pypika import Table

from fathom.constants.storage import SqlParameterStyle
from fathom.infrastructure.interaction.pypika.query import (
    CursorCoordinate,
    CursorPaginatedQuery,
    SortDirection,
    SortOrder,
)


class TestCursorPaginatedQueryAscending(unittest.TestCase):
    """
    Forward (ASCENDING) keyset pagination renders strict `>` boundary predicates.
    """

    def setUp(self) -> None:
        """
        Build an ascending NUMBERED-style helper over the messages table.
        """

        self.__messages: Table = Table("messages")
        self.__helper: CursorPaginatedQuery = CursorPaginatedQuery(
            table=self.__messages,
            ordering=SortOrder(
                column="created_at",
                tiebreaker="id",
                direction=SortDirection.ASCENDING,
            ),
            parameter_style=SqlParameterStyle.NUMBERED,
        )
        self.__helper.where(self.__messages.tenant == self.__helper.bind(value="acme"))

    def test_count_sql_omits_cursor_and_limit(self) -> None:
        """
        COUNT SQL must include filters but neither cursor predicate nor LIMIT.
        """

        count_sql, count_parameters = self.__helper.count_sql_and_parameters()

        self.assertEqual(count_sql, "SELECT COUNT(*) FROM messages WHERE tenant=$1")
        self.assertEqual(count_parameters, ("acme",))
        self.assertNotIn("LIMIT", count_sql)
        self.assertNotIn(">", count_sql)
        self.assertNotIn("<", count_sql)

    def test_page_sql_without_cursor_skips_cursor_predicate(self) -> None:
        """
        With cursor=None the keyset boundary must not appear in page SQL.
        """

        self.__helper.count_sql_and_parameters()

        page_sql, page_parameters = self.__helper.page_sql_and_parameters(cursor=None, limit=25)

        self.assertEqual(
            page_sql,
            "SELECT * FROM messages WHERE tenant=$1 ORDER BY created_at ASC,id ASC LIMIT $2",
        )
        self.assertEqual(page_parameters, ("acme", 25))

    def test_page_sql_with_cursor_appends_keyset_predicate_last(self) -> None:
        """
        ASCENDING cursor must bind exactly three placeholders with `>` operators.
        """

        self.__helper.count_sql_and_parameters()

        page_sql, page_parameters = self.__helper.page_sql_and_parameters(
            cursor=CursorCoordinate(created="2026-01-01T00:00:00+00:00", identifier="msg-1"),
            limit=10,
        )

        self.assertEqual(
            page_sql,
            "SELECT * FROM messages WHERE tenant=$1 AND "
            "(created_at>$2 OR (created_at=$3 AND id>$4)) "
            "ORDER BY created_at ASC,id ASC LIMIT $5",
        )
        self.assertEqual(
            page_parameters,
            ("acme", "2026-01-01T00:00:00+00:00", "2026-01-01T00:00:00+00:00", "msg-1", 10),
        )


class TestCursorPaginatedQueryDescending(unittest.TestCase):
    """
    Reverse (DESCENDING) keyset pagination renders strict `<` boundary predicates.
    """

    def test_descending_uses_less_than_operator_and_order_by(self) -> None:
        """
        DESCENDING direction must flip both the cursor operator and ORDER BY.
        """

        threads = Table("threads")
        helper = CursorPaginatedQuery(
            table=threads,
            ordering=SortOrder(
                column="updated_at",
                tiebreaker="id",
                direction=SortDirection.DESCENDING,
            ),
            parameter_style=SqlParameterStyle.NUMBERED,
        )
        helper.where(threads.tenant == helper.bind(value="acme"))
        helper.count_sql_and_parameters()

        page_sql, page_parameters = helper.page_sql_and_parameters(
            cursor=CursorCoordinate(created="2026-01-01T00:00:00+00:00", identifier="thread-1"),
            limit=5,
        )

        self.assertIn("(updated_at<$2 OR (updated_at=$3 AND id<$4))", page_sql)
        self.assertIn("ORDER BY updated_at DESC,id DESC", page_sql)
        self.assertEqual(
            page_parameters,
            (
                "acme",
                "2026-01-01T00:00:00+00:00",
                "2026-01-01T00:00:00+00:00",
                "thread-1",
                5,
            ),
        )


class TestCursorPaginatedQueryQuestionMarkStyle(unittest.TestCase):
    """
    SQLite (`?`) placeholders must match parameter counts for every case.
    """

    def test_question_mark_count_and_page_placeholder_counts_match_parameters(self) -> None:
        """
        Question-mark placeholders must equal the parameter tuple length.
        """

        events = Table("events")
        helper = CursorPaginatedQuery(
            table=events,
            ordering=SortOrder(
                column="created_at",
                tiebreaker="id",
                direction=SortDirection.ASCENDING,
            ),
            parameter_style=SqlParameterStyle.QUESTION_MARK,
        )
        helper.where(events.tenant == helper.bind(value="acme"))
        helper.where(events.thread == helper.bind(value="thread-1"))
        helper.where(events.kind.isin([helper.bind(value=k) for k in ("a", "b", "c")]))

        count_sql, count_parameters = helper.count_sql_and_parameters()
        page_sql, page_parameters = helper.page_sql_and_parameters(
            cursor=CursorCoordinate(created="2026-01-01T00:00:00+00:00", identifier="e-1"),
            limit=50,
        )

        self.assertEqual(count_sql.count("?"), len(count_parameters))
        self.assertEqual(page_sql.count("?"), len(page_parameters))

    def test_numbered_style_renders_monotonic_dollar_placeholders(self) -> None:
        """
        Numbered placeholders must increment monotonically across filters and cursor.
        """

        events = Table("events")
        helper = CursorPaginatedQuery(
            table=events,
            ordering=SortOrder(
                column="created_at",
                tiebreaker="id",
                direction=SortDirection.ASCENDING,
            ),
            parameter_style=SqlParameterStyle.NUMBERED,
        )
        helper.where(events.tenant == helper.bind(value="acme"))
        helper.where(events.thread == helper.bind(value="thread-1"))
        helper.where(events.kind.isin([helper.bind(value=k) for k in ("a", "b")]))

        count_sql, count_parameters = helper.count_sql_and_parameters()
        page_sql, page_parameters = helper.page_sql_and_parameters(
            cursor=CursorCoordinate(created="2026-01-01T00:00:00+00:00", identifier="e-1"),
            limit=10,
        )

        self.assertEqual(count_sql.count("$"), 4)
        self.assertEqual(len(count_parameters), 4)
        self.assertEqual(page_sql.count("$"), 4 + 3 + 1)
        self.assertEqual(len(page_parameters), 4 + 3 + 1)


class TestCursorPaginatedQueryCountIsolation(unittest.TestCase):
    """
    Count parameters must never leak cursor or limit values.
    """

    def test_count_parameters_snapshot_excludes_subsequent_cursor_bindings(self) -> None:
        """
        Calling page_sql_and_parameters after count must not enlarge the count tuple.
        """

        messages = Table("messages")
        helper = CursorPaginatedQuery(
            table=messages,
            ordering=SortOrder(
                column="created_at",
                tiebreaker="id",
                direction=SortDirection.ASCENDING,
            ),
            parameter_style=SqlParameterStyle.NUMBERED,
        )
        helper.where(messages.tenant == helper.bind(value="acme"))

        count_sql, count_parameters = helper.count_sql_and_parameters()
        helper.page_sql_and_parameters(
            cursor=CursorCoordinate(created="2026-01-01T00:00:00+00:00", identifier="msg-1"),
            limit=10,
        )

        recount_sql, recount_parameters = helper.count_sql_and_parameters()

        self.assertEqual(count_parameters, ("acme",))
        self.assertEqual(count_parameters, recount_parameters)
        self.assertEqual(count_sql, recount_sql)


class TestSortOrderValidation(unittest.TestCase):
    """
    SortOrder must enforce non-empty column/tiebreaker invariants.
    """

    def test_sort_order_rejects_empty_column(self) -> None:
        """
        Pydantic must reject an empty ordering column.
        """

        with self.assertRaises(ValueError):
            SortOrder(column="", tiebreaker="id", direction=SortDirection.ASCENDING)

    def test_sort_order_rejects_empty_tiebreaker(self) -> None:
        """
        Pydantic must reject an empty tiebreaker column.
        """

        with self.assertRaises(ValueError):
            SortOrder(column="created_at", tiebreaker="", direction=SortDirection.ASCENDING)


class TestCursorCoordinateAcceptsBackendShapes(unittest.TestCase):
    """
    CursorCoordinate accepts the backend-shaped timestamp (datetime or string).
    """

    def test_cursor_accepts_isoformat_string(self) -> None:
        """
        SQLite ISO strings are first-class boundary values.
        """

        coordinate = CursorCoordinate(created="2026-01-01T00:00:00+00:00", identifier="id-1")

        self.assertEqual(coordinate.created, "2026-01-01T00:00:00+00:00")
        self.assertEqual(coordinate.identifier, "id-1")

    def test_cursor_accepts_aware_datetime(self) -> None:
        """
        Postgres-shaped datetime values are first-class boundary values.
        """

        moment = datetime(2026, 1, 1, tzinfo=timezone.utc)
        coordinate = CursorCoordinate(created=moment, identifier="id-1")

        self.assertEqual(coordinate.created, moment)
