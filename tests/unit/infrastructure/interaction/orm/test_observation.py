from __future__ import annotations

import logging
from typing import List, Mapping, Optional, Sequence, Tuple
from uuid import uuid4

from fathom.infrastructure.interaction.orm.observation import QueryObserver
from fathom.schemas.sql import SqlParameterValue


class RecordingHandler(logging.Handler):
    """
    Captures log records emitted by query observation tests.
    """

    def __init__(self) -> None:
        """
        Initialize the in-memory record buffer.
        """

        super().__init__()
        self.records: List[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        """
        Store one log record.
        """

        self.records.append(record)


class Clock:
    """
    Returns deterministic timestamps for duration assertions.
    """

    def __init__(self, *, ticks: Tuple[float, ...]) -> None:
        """
        Store the ordered timestamp values.
        """

        self.__ticks = list(ticks)

    def __call__(self) -> float:
        """
        Return the next timestamp value.
        """

        return self.__ticks.pop(0)


class ObservedClient:
    """
    Test double for the Tortoise database client execution surface.
    """

    def __init__(self) -> None:
        """
        Initialize captured calls.
        """

        self.calls: List[str] = []

    async def execute_query(
        self,
        query: str,
        values: Optional[List[SqlParameterValue]] = None,
    ) -> Tuple[int, Sequence[Mapping[str, object]]]:
        """
        Capture one row-returning query call.
        """

        self.calls.append(f"query:{query}:{values}")
        return (1, ())

    async def execute_query_dict(
        self,
        query: str,
        values: Optional[List[SqlParameterValue]] = None,
    ) -> List[Mapping[str, object]]:
        """
        Capture one dictionary-row query call.
        """

        self.calls.append(f"dict:{query}:{values}")
        return []

    async def execute_insert(self, query: str, values: List[SqlParameterValue]) -> object:
        """
        Capture one insert call.
        """

        self.calls.append(f"insert:{query}:{values}")
        return object()

    async def execute_many(self, query: str, values: List[List[SqlParameterValue]]) -> None:
        """
        Capture one batch call.
        """

        self.calls.append(f"many:{query}:{values}")

    async def execute_script(self, query: str) -> None:
        """
        Capture one script call.
        """

        self.calls.append(f"script:{query}")


class TestQueryObserver:
    """
    Verify slow-query observation around the ORM database client.
    """

    async def test_slow_query_logs_structured_context(self) -> None:
        """
        Slow observed queries emit operation, duration, and SQL prefix.
        """

        logger, handler = self.__logger()
        observer = QueryObserver(logger=logger, threshold=100, clock=Clock(ticks=(1.0, 1.25)))
        client = ObservedClient()

        observer.observe(client=client)
        await client.execute_query(
            "SELECT *\nFROM conversations WHERE tenant_id = $1", ["tenant-a"]
        )

        assert len(handler.records) == 1
        record = handler.records[0]
        assert record.getMessage() == "Slow ORM query"
        assert record.__dict__["operation"] == "interaction.orm.query"
        assert record.__dict__["duration_ms"] == 250
        assert record.__dict__["sql_head"] == "SELECT * FROM conversations WHERE tenant_id = $1"

    async def test_fast_query_does_not_log(self) -> None:
        """
        Queries below the configured threshold stay quiet.
        """

        logger, handler = self.__logger()
        observer = QueryObserver(logger=logger, threshold=100, clock=Clock(ticks=(1.0, 1.01)))
        client = ObservedClient()

        observer.observe(client=client)
        await client.execute_query_dict("SELECT 1", [])

        assert handler.records == []

    async def test_observe_is_idempotent(self) -> None:
        """
        Observing the same client twice does not double-log one query.
        """

        logger, handler = self.__logger()
        observer = QueryObserver(logger=logger, threshold=0, clock=Clock(ticks=(1.0, 1.02)))
        client = ObservedClient()

        observer.observe(client=client)
        observer.observe(client=client)
        await client.execute_script("SELECT 1")

        assert len(handler.records) == 1
        assert handler.records[0].__dict__["operation"] == "interaction.orm.script"

    def __logger(self) -> Tuple[logging.Logger, RecordingHandler]:
        """
        Build one isolated logger and capture handler.
        """

        logger = logging.getLogger(f"orm-query-observer-test-{uuid4()}")
        handler = RecordingHandler()
        logger.setLevel(logging.WARNING)
        logger.propagate = False
        logger.addHandler(handler)
        return logger, handler
