from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Mapping, Optional, Sequence, Tuple
from uuid import uuid4

from fathom.infrastructure.interaction.orm.raw import RawSql
from fathom.schemas.sql import SqlParameterValue

if TYPE_CHECKING:
    from pathlib import Path


class _RecordingHandler(logging.Handler):
    """
    Captures log records emitted by raw SQL tests.
    """

    def __init__(self) -> None:
        """
        Initialize the captured record list.
        """

        super().__init__()
        self.records: List[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        """
        Store one emitted log record.
        """

        self.records.append(record)


class _StepClock:
    """
    Provides deterministic timestamps for duration tests.
    """

    def __init__(self, *, values: Tuple[float, ...]) -> None:
        """
        Capture timestamps returned in call order.
        """

        self.__values = list(values)

    def __call__(self) -> float:
        """
        Return the next configured timestamp.
        """

        return self.__values.pop(0)


class _FakeRawConnection:
    """
    Fake asyncpg-like connection for raw SQL executor tests.
    """

    def __init__(self) -> None:
        """
        Initialize captured calls and returned rows.
        """

        self.calls: List[Tuple[str, str, Tuple[SqlParameterValue, ...]]] = []
        self.rows: List[Mapping[str, object]] = [{"id": "row"}]
        self.row: Optional[Mapping[str, object]] = {"id": "one"}

    async def execute(self, query: str, *args: SqlParameterValue) -> str:
        """
        Capture an execute call.
        """

        self.calls.append(("execute", query, args))
        return "OK"

    async def fetch(self, query: str, *args: SqlParameterValue) -> Sequence[Mapping[str, object]]:
        """
        Capture a fetch call.
        """

        self.calls.append(("fetch", query, args))
        return self.rows

    async def fetchrow(
        self, query: str, *args: SqlParameterValue
    ) -> Optional[Mapping[str, object]]:
        """
        Capture a fetchrow call.
        """

        self.calls.append(("fetchrow", query, args))
        return self.row


class TestRawSql:
    """
    Verify named SQL-file execution behavior.
    """

    def test_compile_reuses_repeated_named_parameter(self, tmp_path: Path) -> None:
        (tmp_path / "query.sql").write_text(
            "SELECT * FROM jobs WHERE tenant = :tenant OR owner = :tenant",
            encoding="utf-8",
        )
        raw = RawSql(root=tmp_path)

        compiled = raw.compile(name="query.sql")

        assert compiled.statement == "SELECT * FROM jobs WHERE tenant = $1 OR owner = $1"
        assert compiled.parameters == ("tenant",)
        assert compiled.bind(values={"tenant": "acme"}) == ("acme",)

    def test_compile_caches_named_sql_file(self, tmp_path: Path) -> None:
        (tmp_path / "query.sql").write_text("SELECT :tenant", encoding="utf-8")
        raw = RawSql(root=tmp_path)
        first = raw.compile(name="query.sql")
        (tmp_path / "query.sql").write_text("SELECT :tenant, :other", encoding="utf-8")

        second = raw.compile(name="query.sql")

        assert second is first
        assert second.statement == "SELECT $1"

    def test_compile_ignores_postgres_cast_syntax(self, tmp_path: Path) -> None:
        (tmp_path / "query.sql").write_text(
            "SELECT :value::text AS value",
            encoding="utf-8",
        )
        raw = RawSql(root=tmp_path)

        compiled = raw.compile(name="query.sql")

        assert compiled.statement == "SELECT $1::text AS value"
        assert compiled.parameters == ("value",)

    def test_bind_rejects_missing_parameter(self, tmp_path: Path) -> None:
        (tmp_path / "query.sql").write_text("SELECT :tenant", encoding="utf-8")
        raw = RawSql(root=tmp_path)

        compiled = raw.compile(name="query.sql")

        try:
            compiled.bind(values={})
        except ValueError as exception:
            assert "Missing SQL parameter(s): tenant" in str(exception)
        else:
            raise AssertionError("Expected missing SQL parameter to fail.")

    def test_bind_rejects_unused_parameter(self, tmp_path: Path) -> None:
        (tmp_path / "query.sql").write_text("SELECT :tenant", encoding="utf-8")
        raw = RawSql(root=tmp_path)

        compiled = raw.compile(name="query.sql")

        try:
            compiled.bind(values={"tenant": "acme", "owner": "worker"})
        except ValueError as exception:
            assert "Unused SQL parameter(s): owner" in str(exception)
        else:
            raise AssertionError("Expected unused SQL parameter to fail.")

    async def test_fetch_binds_keyword_arguments(self, tmp_path: Path) -> None:
        (tmp_path / "query.sql").write_text(
            "SELECT * FROM jobs WHERE tenant = :tenant AND owner = :owner",
            encoding="utf-8",
        )
        raw = RawSql(root=tmp_path)
        connection = _FakeRawConnection()

        rows = await raw.fetch(
            connection=connection,
            name="query.sql",
            tenant="acme",
            owner="worker",
        )

        assert rows == [{"id": "row"}]
        assert connection.calls == [
            (
                "fetch",
                "SELECT * FROM jobs WHERE tenant = $1 AND owner = $2",
                ("acme", "worker"),
            )
        ]

    async def test_fetchrow_binds_keyword_arguments(self, tmp_path: Path) -> None:
        (tmp_path / "query.sql").write_text("SELECT :tenant", encoding="utf-8")
        raw = RawSql(root=tmp_path)
        connection = _FakeRawConnection()

        row = await raw.fetchrow(
            connection=connection,
            name="query.sql",
            tenant="acme",
        )

        assert row == {"id": "one"}
        assert connection.calls == [("fetchrow", "SELECT $1", ("acme",))]

    async def test_execute_binds_keyword_arguments(self, tmp_path: Path) -> None:
        (tmp_path / "query.sql").write_text(
            "DELETE FROM jobs WHERE tenant = :tenant", encoding="utf-8"
        )
        raw = RawSql(root=tmp_path)
        connection = _FakeRawConnection()

        result = await raw.execute(
            connection=connection,
            name="query.sql",
            tenant="acme",
        )

        assert result == "OK"
        assert connection.calls == [("execute", "DELETE FROM jobs WHERE tenant = $1", ("acme",))]

    async def test_fetch_logs_slow_query(self, tmp_path: Path) -> None:
        (tmp_path / "query.sql").write_text("SELECT :tenant", encoding="utf-8")
        logger = logging.getLogger(f"raw-sql-test-{uuid4()}")
        logger.setLevel(logging.WARNING)
        logger.propagate = False
        handler = _RecordingHandler()
        logger.addHandler(handler)
        raw = RawSql(
            root=tmp_path,
            logger=logger,
            slow_query_limit=100,
            clock=_StepClock(values=(0.0, 0.25)),
        )
        connection = _FakeRawConnection()

        await raw.fetch(connection=connection, name="query.sql", tenant="acme")

        assert len(handler.records) == 1
        record = handler.records[0]
        assert record.getMessage() == "Slow raw SQL query"
        assert record.__dict__["operation"] == "interaction.raw_sql.slow"
        assert record.__dict__["sql_name"] == "query.sql"
        assert record.__dict__["duration_ms"] == 250

    async def test_fetch_does_not_log_fast_query(self, tmp_path: Path) -> None:
        (tmp_path / "query.sql").write_text("SELECT :tenant", encoding="utf-8")
        logger = logging.getLogger(f"raw-sql-test-{uuid4()}")
        logger.setLevel(logging.WARNING)
        logger.propagate = False
        handler = _RecordingHandler()
        logger.addHandler(handler)
        raw = RawSql(
            root=tmp_path,
            logger=logger,
            slow_query_limit=100,
            clock=_StepClock(values=(0.0, 0.02)),
        )
        connection = _FakeRawConnection()

        await raw.fetch(connection=connection, name="query.sql", tenant="acme")

        assert handler.records == []

    def test_resolve_rejects_missing_file(self, tmp_path: Path) -> None:
        raw = RawSql(root=tmp_path)

        try:
            raw.compile(name="missing.sql")
        except FileNotFoundError as exception:
            assert "SQL file not found: missing.sql" in str(exception)
        else:
            raise AssertionError("Expected missing SQL file to fail.")
