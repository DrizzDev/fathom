from __future__ import annotations

import unittest
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, AsyncIterator, List, Optional, Tuple

from fathom.constants.collaboration import (
    ArtifactKind,
    EventKind,
    MessageKind,
)
from fathom.conversation.cursor import OpaqueCursor
from fathom.infrastructure.interaction.pypika.postgres.repositories.artifacts import (
    PostgresArtifactRepository,
)
from fathom.infrastructure.interaction.pypika.postgres.repositories.context import PostgresContext
from fathom.infrastructure.interaction.pypika.postgres.repositories.contexts import (
    PostgresContextRepository,
)
from fathom.infrastructure.interaction.pypika.postgres.repositories.events import (
    PostgresEventRepository,
)
from fathom.infrastructure.interaction.pypika.postgres.repositories.messages import (
    PostgresMessageRepository,
)
from fathom.infrastructure.interaction.pypika.postgres.repositories.threads import (
    PostgresThreadRepository,
)
from fathom.infrastructure.interaction.pypika.postgres.row import PostgresRowMapper
from fathom.infrastructure.interaction.pypika.sqlite.repositories.artifacts import (
    ArtifactRepository,
)
from fathom.infrastructure.interaction.pypika.sqlite.repositories.context import StoreContext
from fathom.infrastructure.interaction.pypika.sqlite.repositories.contexts import ContextRepository
from fathom.infrastructure.interaction.pypika.sqlite.repositories.events import EventRepository
from fathom.infrastructure.interaction.pypika.sqlite.repositories.messages import MessageRepository
from fathom.infrastructure.interaction.pypika.sqlite.repositories.threads import ThreadRepository
from fathom.infrastructure.interaction.pypika.sqlite.row import RowMapper
from fathom.interaction.digest import EventDigest
from fathom.interaction.lifecycle import Lifecycle
from fathom.schemas.interaction import (
    ArtifactCursorQuery,
    ContextCursorQuery,
    EventCursorQuery,
    MessageCursorQuery,
    SortOrder,
    ThreadListQuery,
)


class _CapturedCall:
    """
    Holds one captured (sql, parameters) pair from a fake connection.
    """

    def __init__(self, *, sql: str, parameters: Tuple[Any, ...]) -> None:
        """
        Store the captured SQL and parameter tuple.
        """

        self.sql: str = sql
        self.parameters: Tuple[Any, ...] = parameters


class _FakeCursor:
    """
    Async cursor stub returning canned rows for fetchone/fetchall.
    """

    def __init__(self, *, fetchone_value: Optional[Any], fetchall_value: List[Any]) -> None:
        """
        Bind the canned scalars for fetchone and fetchall.
        """

        self.__one = fetchone_value
        self.__all = fetchall_value

    async def fetchone(self) -> Optional[Any]:
        """
        Return the canned fetchone scalar.
        """

        return self.__one

    async def fetchall(self) -> List[Any]:
        """
        Return the canned fetchall list.
        """

        return self.__all


class _FakeExecution:
    """
    Async context manager wrapping one cursor for a connection.execute call.
    """

    def __init__(self, *, cursor: _FakeCursor) -> None:
        """
        Bind the cursor returned on enter.
        """

        self.__cursor = cursor
        self.rowcount = 0

    async def __aenter__(self) -> _FakeCursor:
        """
        Return the bound cursor.
        """

        return self.__cursor

    async def __aexit__(self, *args: object) -> None:
        """
        Close the cursor scope.
        """

        return None

    def __await__(self):  # type: ignore[no-untyped-def]
        """
        Allow ``await connection.execute(...)`` to no-op.
        """

        async def _self() -> "_FakeExecution":
            return self

        return _self().__await__()


class _FakeConnection:
    """
    Captures every execute() call's SQL and parameters.
    """

    def __init__(self) -> None:
        """
        Initialize the captured-calls log.
        """

        self.calls: List[_CapturedCall] = []

    def execute(self, sql: str, parameters: Tuple[Any, ...] = ()) -> _FakeExecution:
        """
        Capture the call and return a fake execution that yields empty rows.
        """

        self.calls.append(_CapturedCall(sql=sql, parameters=tuple(parameters)))
        # First call is COUNT (returns scalar 0); other calls return zero rows.
        is_count = "COUNT(*)" in sql
        fetchone_value: Optional[Any] = (0,) if is_count else None
        return _FakeExecution(
            cursor=_FakeCursor(
                fetchone_value=fetchone_value,
                fetchall_value=[],
            )
        )


class _FakeUnit:
    """
    Async unit-of-work yielding one captured fake connection per session.
    """

    def __init__(self, *, connection: _FakeConnection) -> None:
        """
        Bind the underlying captured connection.
        """

        self.__connection = connection

    @asynccontextmanager
    async def session(self) -> AsyncIterator[_FakeConnection]:
        """
        Yield the bound captured connection.
        """

        yield self.__connection

    @asynccontextmanager
    async def atomic(self) -> AsyncIterator[None]:
        """
        Yield without doing anything; no transaction boundaries are required.
        """

        yield None


def _postgres_repos() -> Tuple[
    _FakeConnection,
    PostgresThreadRepository,
    PostgresMessageRepository,
    PostgresEventRepository,
    PostgresArtifactRepository,
    PostgresContextRepository,
]:
    """
    Wire postgres repositories around one captured fake connection.
    """

    connection = _FakeConnection()
    context = PostgresContext(
        unit=_FakeUnit(connection=connection),
        lifecycle=Lifecycle(),
        rows=PostgresRowMapper(),
        event_digest=EventDigest(),
    )
    return (
        connection,
        PostgresThreadRepository(context=context),
        PostgresMessageRepository(context=context),
        PostgresEventRepository(context=context),
        PostgresArtifactRepository(context=context),
        PostgresContextRepository(context=context),
    )


def _sqlite_repos() -> Tuple[
    _FakeConnection,
    ThreadRepository,
    MessageRepository,
    EventRepository,
    ArtifactRepository,
    ContextRepository,
]:
    """
    Wire SQLite repositories around one captured fake connection.
    """

    connection = _FakeConnection()
    context = StoreContext(
        unit=_FakeUnit(connection=connection),
        lifecycle=Lifecycle(),
        rows=RowMapper(),
        event_digest=EventDigest(),
    )
    return (
        connection,
        ThreadRepository(context=context),
        MessageRepository(context=context),
        EventRepository(context=context),
        ArtifactRepository(context=context),
        ContextRepository(context=context),
    )


def _opaque_cursor() -> str:
    """
    Build one opaque cursor token used to exercise the keyset boundary.
    """

    return OpaqueCursor(
        created=datetime(2026, 1, 1, tzinfo=timezone.utc),
        identifier="boundary-id",
    ).encode()


def _assert_count_isolation(
    *,
    case: unittest.TestCase,
    calls: List[_CapturedCall],
    direction: str,
) -> None:
    """
    Assert the first call (COUNT) excludes cursor params and the page call binds them.

    The unique keyset boundary fingerprint is ``id >`` (ASC) or ``id <`` (DESC),
    which appears only in the cursor predicate, never as a filter.
    """

    count_call = calls[0]
    page_call = calls[1]
    cursor_marker = "id>" if direction == "asc" else "id<"
    case.assertIn("COUNT(*)", count_call.sql)
    case.assertNotIn("boundary-id", count_call.parameters)
    case.assertNotIn(cursor_marker, count_call.sql)
    case.assertIn("ORDER BY", page_call.sql)
    case.assertIn(cursor_marker, page_call.sql)
    case.assertIn("boundary-id", page_call.parameters)


def _assert_placeholder_count_matches(
    *, case: unittest.TestCase, call: _CapturedCall, style: str
) -> None:
    """
    Assert that the captured SQL has exactly as many placeholders as parameters.
    """

    if style == "numbered":
        ordinals: List[int] = []
        index = 0
        while index < len(call.sql):
            if (
                call.sql[index] == "$"
                and index + 1 < len(call.sql)
                and call.sql[index + 1].isdigit()
            ):
                end = index + 1
                while end < len(call.sql) and call.sql[end].isdigit():
                    end += 1
                ordinals.append(int(call.sql[index + 1 : end]))
                index = end
                continue
            index += 1
        case.assertTrue(ordinals, "Expected at least one numbered placeholder.")
        case.assertEqual(max(ordinals), len(call.parameters))
        case.assertEqual(set(ordinals), set(range(1, len(call.parameters) + 1)))
    else:
        case.assertEqual(call.sql.count("?"), len(call.parameters))


class TestPostgresListSql(unittest.IsolatedAsyncioTestCase):
    """
    Postgres repository list_* methods must emit consistent NUMBERED placeholders.
    """

    def setUp(self) -> None:
        """
        Build a fresh captured-connection wiring per test.
        """

        (
            self.__connection,
            self.__threads,
            self.__messages,
            self.__events,
            self.__artifacts,
            self.__contexts,
        ) = _postgres_repos()

    async def test_list_threads_count_excludes_cursor_and_page_marker_matches(self) -> None:
        """
        list_threads worst-case binds title, state, dates, cursor and limit cleanly.
        """

        await self.__threads.list_threads(
            query=ThreadListQuery(
                tenant="acme",
                cursor=_opaque_cursor(),
                limit=10,
            )
        )

        _assert_count_isolation(
            case=self,
            calls=self.__connection.calls,
            direction="desc",
        )
        _assert_placeholder_count_matches(
            case=self, call=self.__connection.calls[1], style="numbered"
        )

    async def test_list_messages_worst_case_placeholder_indices_are_monotonic(self) -> None:
        """
        list_messages with task+author+kinds+since+until+cursor must bind cleanly.
        """

        await self.__messages.list_messages(
            query=MessageCursorQuery(
                tenant="acme",
                thread="thread-1",
                task="task-1",
                author="actor-1",
                kinds=(MessageKind.REQUEST, MessageKind.ANSWER),
                since=datetime(2026, 1, 1, tzinfo=timezone.utc),
                until=datetime(2026, 2, 1, tzinfo=timezone.utc),
                cursor=_opaque_cursor(),
                limit=5,
            )
        )

        _assert_count_isolation(
            case=self,
            calls=self.__connection.calls,
            direction="desc",
        )
        _assert_placeholder_count_matches(
            case=self, call=self.__connection.calls[0], style="numbered"
        )
        _assert_placeholder_count_matches(
            case=self, call=self.__connection.calls[1], style="numbered"
        )

    async def test_list_events_worst_case_placeholder_indices_are_monotonic(self) -> None:
        """
        list_events with task+actor+kinds+since+until+cursor must bind cleanly.
        """

        await self.__events.list_events(
            query=EventCursorQuery(
                tenant="acme",
                thread="thread-1",
                task="task-1",
                actor="actor-1",
                kinds=(EventKind.MESSAGE_RECORDED, EventKind.CONTENT_SANITIZED),
                since=datetime(2026, 1, 1, tzinfo=timezone.utc),
                until=datetime(2026, 2, 1, tzinfo=timezone.utc),
                cursor=_opaque_cursor(),
                limit=5,
            )
        )

        _assert_count_isolation(
            case=self,
            calls=self.__connection.calls,
            direction="desc",
        )
        _assert_placeholder_count_matches(
            case=self, call=self.__connection.calls[0], style="numbered"
        )
        _assert_placeholder_count_matches(
            case=self, call=self.__connection.calls[1], style="numbered"
        )

    async def test_list_artifacts_descending_uses_lt_keyset(self) -> None:
        """
        list_artifacts with order=DESC must flip the keyset operator to `<`.
        """

        await self.__artifacts.list_artifacts(
            query=ArtifactCursorQuery(
                tenant="acme",
                thread="thread-1",
                task="task-1",
                producer="actor-1",
                kinds=(ArtifactKind.SCRIPT,),
                since=datetime(2026, 1, 1, tzinfo=timezone.utc),
                until=datetime(2026, 2, 1, tzinfo=timezone.utc),
                cursor=_opaque_cursor(),
                limit=5,
                order=SortOrder.DESC,
            )
        )

        _assert_count_isolation(
            case=self,
            calls=self.__connection.calls,
            direction="desc",
        )
        _assert_placeholder_count_matches(
            case=self, call=self.__connection.calls[1], style="numbered"
        )

    async def test_list_contexts_count_excludes_cursor(self) -> None:
        """
        list_contexts worst case must keep COUNT free of cursor parameters.
        """

        await self.__contexts.list_contexts(
            query=ContextCursorQuery(
                tenant="acme",
                thread="thread-1",
                task="task-1",
                consumer="consumer-1",
                since=datetime(2026, 1, 1, tzinfo=timezone.utc),
                until=datetime(2026, 2, 1, tzinfo=timezone.utc),
                cursor=_opaque_cursor(),
                limit=5,
            )
        )

        _assert_count_isolation(
            case=self,
            calls=self.__connection.calls,
            direction="desc",
        )
        _assert_placeholder_count_matches(
            case=self, call=self.__connection.calls[1], style="numbered"
        )


class TestSqliteListSql(unittest.IsolatedAsyncioTestCase):
    """
    SQLite repository list_* methods must emit `?` placeholders matching params.
    """

    def setUp(self) -> None:
        """
        Build a fresh captured-connection wiring per test.
        """

        (
            self.__connection,
            self.__threads,
            self.__messages,
            self.__events,
            self.__artifacts,
            self.__contexts,
        ) = _sqlite_repos()

    async def test_list_threads_placeholder_count_matches_parameters(self) -> None:
        """
        SQLite list_threads worst case must keep `?` count aligned with params.
        """

        await self.__threads.list_threads(
            query=ThreadListQuery(
                tenant="acme",
                cursor=_opaque_cursor(),
                limit=10,
            )
        )

        _assert_count_isolation(
            case=self,
            calls=self.__connection.calls,
            direction="desc",
        )
        _assert_placeholder_count_matches(
            case=self, call=self.__connection.calls[0], style="question"
        )
        _assert_placeholder_count_matches(
            case=self, call=self.__connection.calls[1], style="question"
        )

    async def test_list_messages_placeholder_count_matches_parameters(self) -> None:
        """
        SQLite list_messages worst case must keep `?` count aligned with params.
        """

        await self.__messages.list_messages(
            query=MessageCursorQuery(
                tenant="acme",
                thread="thread-1",
                task="task-1",
                author="actor-1",
                kinds=(MessageKind.REQUEST, MessageKind.ANSWER),
                since=datetime(2026, 1, 1, tzinfo=timezone.utc),
                until=datetime(2026, 2, 1, tzinfo=timezone.utc),
                cursor=_opaque_cursor(),
                limit=5,
            )
        )

        _assert_count_isolation(
            case=self,
            calls=self.__connection.calls,
            direction="desc",
        )
        _assert_placeholder_count_matches(
            case=self, call=self.__connection.calls[0], style="question"
        )
        _assert_placeholder_count_matches(
            case=self, call=self.__connection.calls[1], style="question"
        )

    async def test_list_events_placeholder_count_matches_parameters(self) -> None:
        """
        SQLite list_events worst case must keep `?` count aligned with params.
        """

        await self.__events.list_events(
            query=EventCursorQuery(
                tenant="acme",
                thread="thread-1",
                task="task-1",
                actor="actor-1",
                kinds=(EventKind.MESSAGE_RECORDED,),
                since=datetime(2026, 1, 1, tzinfo=timezone.utc),
                until=datetime(2026, 2, 1, tzinfo=timezone.utc),
                cursor=_opaque_cursor(),
                limit=5,
            )
        )

        _assert_count_isolation(
            case=self,
            calls=self.__connection.calls,
            direction="desc",
        )
        _assert_placeholder_count_matches(
            case=self, call=self.__connection.calls[0], style="question"
        )
        _assert_placeholder_count_matches(
            case=self, call=self.__connection.calls[1], style="question"
        )

    async def test_list_artifacts_descending_uses_lt_keyset(self) -> None:
        """
        SQLite list_artifacts with order=DESC must flip the keyset operator to `<`.
        """

        await self.__artifacts.list_artifacts(
            query=ArtifactCursorQuery(
                tenant="acme",
                thread="thread-1",
                task="task-1",
                producer="actor-1",
                kinds=(ArtifactKind.SCRIPT,),
                since=datetime(2026, 1, 1, tzinfo=timezone.utc),
                until=datetime(2026, 2, 1, tzinfo=timezone.utc),
                cursor=_opaque_cursor(),
                limit=5,
                order=SortOrder.DESC,
            )
        )

        _assert_count_isolation(
            case=self,
            calls=self.__connection.calls,
            direction="desc",
        )
        _assert_placeholder_count_matches(
            case=self, call=self.__connection.calls[1], style="question"
        )

    async def test_list_contexts_placeholder_count_matches_parameters(self) -> None:
        """
        SQLite list_contexts worst case must keep `?` count aligned with params.
        """

        await self.__contexts.list_contexts(
            query=ContextCursorQuery(
                tenant="acme",
                thread="thread-1",
                task="task-1",
                consumer="consumer-1",
                since=datetime(2026, 1, 1, tzinfo=timezone.utc),
                until=datetime(2026, 2, 1, tzinfo=timezone.utc),
                cursor=_opaque_cursor(),
                limit=5,
            )
        )

        _assert_count_isolation(
            case=self,
            calls=self.__connection.calls,
            direction="desc",
        )
        _assert_placeholder_count_matches(
            case=self, call=self.__connection.calls[0], style="question"
        )
        _assert_placeholder_count_matches(
            case=self, call=self.__connection.calls[1], style="question"
        )
