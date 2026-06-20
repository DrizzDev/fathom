from __future__ import annotations

import unittest
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

from fathom.infrastructure.interaction.pypika.sqlite.repositories.cleanup import CleanupService
from fathom.schemas.interaction import CleanupRequest


class _FakeCursor:
    """
    Minimal cursor stand-in returning a configured list of result rows.
    """

    def __init__(self, *, rows: Tuple[Dict[str, Any], ...]) -> None:
        """
        Bind the rows that fetchall / fetchone will surface.
        """

        self.__rows = list(rows)

    async def fetchall(self) -> List[Dict[str, Any]]:
        """
        Return all configured rows in order.
        """

        return list(self.__rows)

    async def fetchone(self) -> Optional[Dict[str, Any]]:
        """
        Return the first row or None when empty.
        """

        return self.__rows[0] if self.__rows else None


class _FakeExecution:
    """
    Awaitable + async-context wrapper mirroring an aiosqlite cursor.
    """

    def __init__(
        self,
        *,
        rows: Tuple[Dict[str, Any], ...] = (),
        rowcount: int = 0,
    ) -> None:
        """
        Bind the rows surfaced as a cursor and the rowcount surfaced when awaited.
        """

        self.__rows = rows
        self.rowcount = rowcount

    def __await__(self):
        """
        Allow `result = await connection.execute(...)` and expose rowcount.
        """

        async def _resolve() -> "_FakeExecution":
            return self

        return _resolve().__await__()

    async def __aenter__(self) -> _FakeCursor:
        """
        Allow `async with connection.execute(...) as cursor:` for SELECTs.
        """

        return _FakeCursor(rows=self.__rows)

    async def __aexit__(self, *args: object) -> None:
        """
        Leave the async context without suppressing exceptions.
        """

        return None


class _FakeSqliteConnection:
    """
    Recording connection that hands out scripted executions in call order.
    """

    def __init__(self, *, responses: Tuple[_FakeExecution, ...] = ()) -> None:
        """
        Bind the queue of scripted executions and reset the call log.
        """

        self.calls: List[Tuple[str, Tuple[object, ...]]] = []
        self.__responses = list(responses)

    def execute(
        self,
        sql: str,
        parameters: Tuple[object, ...] | List[object] = (),
    ) -> _FakeExecution:
        """
        Capture the SQL/parameter pair and pop the next scripted execution.
        """

        self.calls.append((sql, tuple(parameters)))
        if not self.__responses:
            return _FakeExecution()

        return self.__responses.pop(0)


class _FakeUnit:
    """
    Unit-of-work facade that hands out the configured fake connection.
    """

    def __init__(self, *, connection: _FakeSqliteConnection) -> None:
        """
        Bind the fake connection that session() will yield.
        """

        self.__connection = connection

    @asynccontextmanager
    async def session(self) -> AsyncGenerator[_FakeSqliteConnection, None]:
        """
        Yield the fake connection without opening a real transaction.
        """

        yield self.__connection


class _FakeStoreContext:
    """
    Minimal SQLite context: serialises datetimes and exposes the unit.
    """

    def __init__(self, *, connection: _FakeSqliteConnection) -> None:
        """
        Bind the fake connection so the unit-of-work can yield it.
        """

        self.__unit = _FakeUnit(connection=connection)

    @property
    def unit(self) -> _FakeUnit:
        """
        Expose the fake unit to the cleanup service.
        """

        return self.__unit

    def _time(self, *, value: datetime) -> str:
        """
        Serialise to ISO format mirroring SQLite StoreContext._time.
        """

        return value.isoformat()


class _CleanupServiceTestBase(unittest.IsolatedAsyncioTestCase):
    """
    Shared scaffolding for the SQLite cleanup service test classes.
    """

    def _build_service(
        self,
        *,
        responses: Tuple[_FakeExecution, ...] = (),
    ) -> Tuple[CleanupService, _FakeSqliteConnection]:
        """
        Construct a cleanup service wired to a recording connection.
        """

        connection = _FakeSqliteConnection(responses=responses)
        context = _FakeStoreContext(connection=connection)

        return CleanupService(context=context), connection


class TestCleanupServiceNoOp(_CleanupServiceTestBase):
    """
    Confirms that a request with no thresholds executes no SQL.
    """

    async def test_cleanup_with_all_thresholds_unset_runs_no_sql(self) -> None:
        """
        Every per-scope threshold being None must result in zero execute() calls.
        """

        service, connection = self._build_service()
        request = CleanupRequest()

        result = await service.cleanup(request=request)

        self.assertEqual(connection.calls, [])
        self.assertEqual(result.idempotency_deleted, 0)
        self.assertEqual(result.jobs_deleted, 0)
        self.assertEqual(result.events_deleted, 0)
        self.assertEqual(result.threads_purged, 0)
        self.assertEqual(result.tasks_purged, 0)
        self.assertEqual(result.messages_purged, 0)
        self.assertEqual(result.artifacts_purged, 0)
        self.assertEqual(result.memberships_purged, 0)
        self.assertEqual(result.contexts_purged, 0)
        self.assertEqual(result.jobs_cascade_purged, 0)
        self.assertEqual(result.events_cascade_purged, 0)
        self.assertEqual(result.sequences_purged, 0)


class TestCleanupServiceRequestsScope(_CleanupServiceTestBase):
    """
    Covers the idempotency-requests retention sweep branch.
    """

    async def test_single_statement_tuple_in_with_iso_timestamp(self) -> None:
        """
        One DELETE binds an ISO threshold + tenant + limit through a tuple-IN subquery.
        """

        before = datetime(2026, 1, 1, tzinfo=timezone.utc)
        service, connection = self._build_service(
            responses=(_FakeExecution(rowcount=2),),
        )
        request = CleanupRequest(
            tenant="acme",
            idempotency_before=before,
            limit=500,
        )

        result = await service.cleanup(request=request)

        self.assertEqual(result.idempotency_deleted, 2)
        self.assertEqual(len(connection.calls), 1)
        sql, params = connection.calls[0]
        self.assertEqual(
            sql,
            "DELETE FROM requests WHERE (tenant,key) IN "
            "(SELECT tenant,key FROM requests WHERE expires_at<? AND tenant=? "
            "ORDER BY tenant,key LIMIT ?)",
        )
        self.assertEqual(params, (before.isoformat(), "acme", 500))

    async def test_global_sweep_omits_tenant_predicate(self) -> None:
        """
        With tenant=None the subquery must not bind a tenant placeholder.
        """

        before = datetime(2026, 1, 1, tzinfo=timezone.utc)
        service, connection = self._build_service(
            responses=(_FakeExecution(rowcount=0),),
        )
        request = CleanupRequest(idempotency_before=before, limit=10)

        await service.cleanup(request=request)

        sql, params = connection.calls[0]
        self.assertEqual(
            sql,
            "DELETE FROM requests WHERE (tenant,key) IN "
            "(SELECT tenant,key FROM requests WHERE expires_at<? "
            "ORDER BY tenant,key LIMIT ?)",
        )
        self.assertEqual(params, (before.isoformat(), 10))


class TestCleanupServiceTerminalJobsScope(_CleanupServiceTestBase):
    """
    Covers the terminal-jobs retention sweep branch.
    """

    async def test_select_includes_all_three_terminal_states(self) -> None:
        """
        The IN predicate must bind completed/failed/abandoned in that order.
        """

        before = datetime(2026, 1, 1, tzinfo=timezone.utc)
        service, connection = self._build_service(
            responses=(_FakeExecution(rowcount=1),),
        )
        request = CleanupRequest(
            tenant="acme",
            terminal_jobs_before=before,
            limit=100,
        )

        result = await service.cleanup(request=request)

        self.assertEqual(result.jobs_deleted, 1)
        self.assertEqual(len(connection.calls), 1)
        sql, params = connection.calls[0]
        self.assertEqual(
            sql,
            "DELETE FROM jobs WHERE (tenant,id) IN "
            "(SELECT tenant,id FROM jobs WHERE state IN (?,?,?) AND updated_at<? "
            "AND tenant=? ORDER BY tenant,id LIMIT ?)",
        )
        self.assertEqual(
            params,
            ("completed", "failed", "abandoned", before.isoformat(), "acme", 100),
        )

    async def test_zero_matches_runs_no_delete(self) -> None:
        """
        An empty rowcount must still record exactly one execute() call.
        """

        before = datetime(2026, 1, 1, tzinfo=timezone.utc)
        service, connection = self._build_service(
            responses=(_FakeExecution(rowcount=0),),
        )
        request = CleanupRequest(terminal_jobs_before=before, limit=50)

        result = await service.cleanup(request=request)

        self.assertEqual(result.jobs_deleted, 0)
        self.assertEqual(len(connection.calls), 1)


class TestCleanupServiceEventsScope(_CleanupServiceTestBase):
    """
    Covers the lifecycle-events retention sweep branch.
    """

    async def test_select_filters_on_created_threshold(self) -> None:
        """
        The subquery must filter on `created_at < ?` and the DELETE must use
        primary-key tuple lookup.
        """

        before = datetime(2026, 1, 1, tzinfo=timezone.utc)
        service, connection = self._build_service(
            responses=(_FakeExecution(rowcount=2),),
        )
        request = CleanupRequest(
            tenant="acme",
            events_before=before,
            limit=200,
        )

        result = await service.cleanup(request=request)

        self.assertEqual(result.events_deleted, 2)
        sql, params = connection.calls[0]
        self.assertEqual(
            sql,
            "DELETE FROM events WHERE (tenant,id) IN "
            "(SELECT tenant,id FROM events WHERE created_at<? AND tenant=? "
            "ORDER BY tenant,id LIMIT ?)",
        )
        self.assertEqual(params, (before.isoformat(), "acme", 200))


class TestCleanupServiceSoftDeletedThreads(_CleanupServiceTestBase):
    """
    Covers the soft-deleted thread purge cascade branch.
    """

    async def test_thread_purge_cascades_through_all_six_dependent_tables(self) -> None:
        """
        Each matched thread must trigger script/version cleanup before the
        existing FK-bound cascade order.
        """

        before = datetime(2026, 1, 1, tzinfo=timezone.utc)
        thread_rows = ({"tenant": "acme", "id": "t1"},)
        responses = (
            _FakeExecution(rowcount=0),  # messages purge DELETE
            _FakeExecution(rowcount=0),  # artifacts purge DELETE
            _FakeExecution(rowcount=0),  # tasks purge DELETE
            _FakeExecution(rows=thread_rows),  # threads SELECT
            _FakeExecution(rowcount=7),  # script versions
            _FakeExecution(rowcount=6),  # scripts
            _FakeExecution(rowcount=3),  # memberships
            _FakeExecution(rowcount=2),  # contexts
            _FakeExecution(rowcount=4),  # jobs
            _FakeExecution(rowcount=5),  # events
            _FakeExecution(rowcount=1),  # sequences
            _FakeExecution(rowcount=1),  # threads
        )
        service, connection = self._build_service(responses=responses)
        request = CleanupRequest(
            tenant="acme",
            soft_deleted_before=before,
            limit=10,
        )

        result = await service.cleanup(request=request)

        self.assertEqual(result.threads_purged, 1)
        self.assertEqual(result.memberships_purged, 3)
        self.assertEqual(result.contexts_purged, 2)
        self.assertEqual(result.jobs_cascade_purged, 4)
        self.assertEqual(result.events_cascade_purged, 5)
        self.assertEqual(result.sequences_purged, 1)
        self.assertEqual(result.script_versions_purged, 7)
        self.assertEqual(result.scripts_purged, 6)
        cascade_tables = [
            "script_versions",
            "scripts",
            "memberships",
            "contexts",
            "jobs",
            "events",
            "sequences",
            "threads",
        ]
        cascade_calls = connection.calls[-8:]
        for index, table in enumerate(cascade_tables):
            sql, params = cascade_calls[index]
            self.assertIn(f"DELETE FROM {table}", sql)
            self.assertEqual(params, ("acme", "t1"))

    async def test_thread_purge_select_filters_by_three_not_exists_clauses(self) -> None:
        """
        The thread purge SELECT must exclude threads that still have tasks,
        messages, or artifacts referencing them.
        """

        before = datetime(2026, 1, 1, tzinfo=timezone.utc)
        responses = (
            _FakeExecution(rowcount=0),  # messages purge DELETE
            _FakeExecution(rowcount=0),  # artifacts purge DELETE
            _FakeExecution(rowcount=0),  # tasks purge DELETE
            _FakeExecution(rows=()),  # threads SELECT
        )
        service, connection = self._build_service(responses=responses)
        request = CleanupRequest(
            tenant="acme",
            soft_deleted_before=before,
            limit=10,
        )

        await service.cleanup(request=request)

        threads_select_sql, _ = connection.calls[3]
        self.assertIn("FROM threads target", threads_select_sql)
        self.assertIn("NOT target.deleted_at IS NULL", threads_select_sql)
        self.assertEqual(threads_select_sql.count("NOT EXISTS"), 3)
        self.assertIn("FROM tasks", threads_select_sql)
        self.assertIn("FROM messages", threads_select_sql)
        self.assertIn("FROM artifacts", threads_select_sql)


class TestCleanupServiceSoftDeletedTasks(_CleanupServiceTestBase):
    """
    Covers the soft-deleted task purge branch.
    """

    async def test_task_purge_uses_eight_not_exists_guards_in_one_statement(self) -> None:
        """
        Tasks DELETE must wrap a subquery with eight NOT EXISTS dependency guards.
        """

        before = datetime(2026, 1, 1, tzinfo=timezone.utc)
        responses = (
            _FakeExecution(rowcount=0),  # messages purge DELETE
            _FakeExecution(rowcount=0),  # artifacts purge DELETE
            _FakeExecution(rowcount=1),  # tasks purge DELETE
            _FakeExecution(rows=()),  # threads SELECT (no rows)
        )
        service, connection = self._build_service(responses=responses)
        request = CleanupRequest(
            tenant="acme",
            soft_deleted_before=before,
            limit=10,
        )

        result = await service.cleanup(request=request)

        self.assertEqual(result.tasks_purged, 1)
        tasks_sql, _ = connection.calls[2]
        self.assertTrue(tasks_sql.startswith("DELETE FROM tasks WHERE (tenant,id) IN "))
        self.assertEqual(tasks_sql.count("NOT EXISTS"), 8)


class TestCleanupServiceSoftDeletedMessages(_CleanupServiceTestBase):
    """
    Covers the soft-deleted message purge branch.
    """

    async def test_message_purge_excludes_messages_referenced_by_tasks_or_replies(self) -> None:
        """
        The message purge subquery must exclude messages still referenced by
        tasks (origin) or by other messages (reply chain).
        """

        before = datetime(2026, 1, 1, tzinfo=timezone.utc)
        responses = (
            _FakeExecution(rowcount=2),  # messages purge DELETE
            _FakeExecution(rowcount=0),  # artifacts purge DELETE
            _FakeExecution(rowcount=0),  # tasks purge DELETE
            _FakeExecution(rows=()),  # threads SELECT
        )
        service, connection = self._build_service(responses=responses)
        request = CleanupRequest(
            tenant="acme",
            soft_deleted_before=before,
            limit=10,
        )

        result = await service.cleanup(request=request)

        self.assertEqual(result.messages_purged, 2)
        messages_sql, _ = connection.calls[0]
        self.assertTrue(messages_sql.startswith("DELETE FROM messages WHERE (tenant,id) IN "))
        self.assertEqual(messages_sql.count("NOT EXISTS"), 2)
        self.assertIn("FROM tasks", messages_sql)
        self.assertIn("FROM messages child", messages_sql)


class TestCleanupServiceSoftDeletedArtifacts(_CleanupServiceTestBase):
    """
    Covers the soft-deleted artifact purge branch and table allowlist.
    """

    async def test_artifact_purge_single_statement_tuple_in(self) -> None:
        """
        The artifacts purge must render one DELETE with a tuple-IN subquery
        that filters out artifacts still referenced by scripts/script_versions.
        """

        before = datetime(2026, 1, 1, tzinfo=timezone.utc)
        responses = (
            _FakeExecution(rowcount=0),  # messages purge DELETE
            _FakeExecution(rowcount=1),  # artifacts purge DELETE
            _FakeExecution(rowcount=0),  # tasks purge DELETE
            _FakeExecution(rows=()),  # threads SELECT
        )
        service, connection = self._build_service(responses=responses)
        request = CleanupRequest(
            tenant="acme",
            soft_deleted_before=before,
            limit=10,
        )

        result = await service.cleanup(request=request)

        self.assertEqual(result.artifacts_purged, 1)
        artifacts_sql, artifacts_params = connection.calls[1]
        self.assertEqual(
            artifacts_sql,
            "DELETE FROM artifacts WHERE (tenant,id) IN "
            "(SELECT target.tenant,target.id FROM artifacts target "
            "WHERE NOT target.deleted_at IS NULL AND target.deleted_at<? "
            "AND NOT EXISTS (SELECT 1 FROM scripts "
            "WHERE (scripts.tenant=target.tenant AND scripts.artifact=target.id)) "
            "AND NOT EXISTS (SELECT 1 FROM script_versions "
            "WHERE (script_versions.tenant=target.tenant "
            "AND script_versions.artifact=target.id)) "
            "AND target.tenant=? ORDER BY target.tenant,target.id LIMIT ?)",
        )
        self.assertEqual(artifacts_params, (before.isoformat(), "acme", 10))


class TestCleanupServiceCombinedScopes(_CleanupServiceTestBase):
    """
    Covers a request that exercises every scope together.
    """

    async def test_all_scopes_aggregate_into_cleanup_result(self) -> None:
        """
        A request with every threshold set must populate every counter on the
        returned CleanupResult.
        """

        before = datetime(2026, 1, 1, tzinfo=timezone.utc)
        responses = (
            _FakeExecution(rowcount=1),  # requests DELETE
            _FakeExecution(rowcount=1),  # jobs DELETE
            _FakeExecution(rowcount=1),  # events DELETE
            _FakeExecution(rowcount=1),  # messages purge DELETE
            _FakeExecution(rowcount=1),  # artifacts purge DELETE
            _FakeExecution(rowcount=1),  # tasks purge DELETE
            _FakeExecution(rows=({"tenant": "acme", "id": "thread1"},)),  # threads SELECT
            _FakeExecution(rowcount=5),  # script versions
            _FakeExecution(rowcount=2),  # scripts
            _FakeExecution(rowcount=2),  # memberships
            _FakeExecution(rowcount=1),  # contexts
            _FakeExecution(rowcount=3),  # cascade jobs
            _FakeExecution(rowcount=4),  # cascade events
            _FakeExecution(rowcount=1),  # sequences
            _FakeExecution(rowcount=1),  # threads
        )
        service, _ = self._build_service(responses=responses)
        request = CleanupRequest(
            tenant="acme",
            idempotency_before=before,
            terminal_jobs_before=before,
            events_before=before,
            soft_deleted_before=before,
            limit=10,
        )

        result = await service.cleanup(request=request)

        self.assertEqual(result.idempotency_deleted, 1)
        self.assertEqual(result.jobs_deleted, 1)
        self.assertEqual(result.events_deleted, 1)
        self.assertEqual(result.messages_purged, 1)
        self.assertEqual(result.artifacts_purged, 1)
        self.assertEqual(result.tasks_purged, 1)
        self.assertEqual(result.threads_purged, 1)
        self.assertEqual(result.memberships_purged, 2)
        self.assertEqual(result.contexts_purged, 1)
        self.assertEqual(result.jobs_cascade_purged, 3)
        self.assertEqual(result.events_cascade_purged, 4)
        self.assertEqual(result.sequences_purged, 1)
        self.assertEqual(result.script_versions_purged, 5)
        self.assertEqual(result.scripts_purged, 2)


class TestCleanupServiceInjectionSafety(_CleanupServiceTestBase):
    """
    Confirms that hostile tenant strings stay parameterized on SQLite.
    """

    async def test_tenant_string_with_quotes_lands_in_parameters_only(self) -> None:
        """
        A tenant containing SQL meta-characters must be bound, never
        interpolated into the SQL fragment.
        """

        before = datetime(2026, 1, 1, tzinfo=timezone.utc)
        attack = "x'; DROP TABLE jobs;--"
        service, connection = self._build_service(
            responses=(_FakeExecution(rowcount=0),),
        )
        request = CleanupRequest(
            tenant=attack,
            terminal_jobs_before=before,
            limit=10,
        )

        await service.cleanup(request=request)

        sql, params = connection.calls[0]
        self.assertNotIn("DROP TABLE", sql)
        self.assertNotIn(attack, sql)
        self.assertIn(attack, params)
