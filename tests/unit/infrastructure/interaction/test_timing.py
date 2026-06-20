from __future__ import annotations

import asyncio
import time
import unittest
from logging import getLogger
from typing import Any, List
from unittest.mock import MagicMock

from fathom.constants.events import FathomEvent
from fathom.infrastructure.interaction.timing import SlowQueryLogger, TimedConnection


class _FakeCursor:
    """
    Minimal async-context cursor used to drive TimedConnection through async-with paths.
    """

    def __init__(self, *, delay_seconds: float) -> None:
        """
        Bind the artificial delay that simulates query execution time.
        """

        self.__delay = delay_seconds

    async def __aenter__(self) -> "_FakeCursor":
        """
        Sleep to simulate the query body, then return self for fetch calls.
        """

        await asyncio.sleep(self.__delay)
        return self

    async def __aexit__(self, *args: object) -> None:
        """
        Close the cursor without further work.
        """

        return None


class _FakeAwaitable:
    """
    Awaitable wrapper that resolves to a fixed rowcount after an artificial delay.
    """

    def __init__(self, *, delay_seconds: float, rowcount: int) -> None:
        """
        Bind the artificial delay and pre-baked rowcount used on await.
        """

        self.__delay = delay_seconds
        self.rowcount = rowcount

    def __await__(self) -> Any:
        """
        Resolve into self after the configured artificial delay.
        """

        async def _resolve() -> "_FakeAwaitable":
            await asyncio.sleep(self.__delay)
            return self

        return _resolve().__await__()


class _FakeConnection:
    """
    Connection stand-in returning either an awaitable or async-context per call mode.
    """

    def __init__(self, *, delay_seconds: float, mode: str) -> None:
        """
        Bind execution delay and the call mode the next execute() call should return.
        """

        self.__delay = delay_seconds
        self.__mode = mode

    def execute(self, sql: str, *parameters: object) -> Any:  # noqa: ARG002
        """
        Return either an awaitable execution or an async-context cursor.
        """

        if self.__mode == "await":
            return _FakeAwaitable(delay_seconds=self.__delay, rowcount=1)
        return _FakeCursor(delay_seconds=self.__delay)


class TestSlowQueryLogger(unittest.TestCase):
    """
    SlowQueryLogger emits one warning per breach and stays silent when disabled.
    """

    def setUp(self) -> None:
        """
        Wire a logger spy that captures structured warnings for assertion.
        """

        self.__warnings: List[tuple] = []
        logger = getLogger("fathom.interaction.test.timing")
        logger.warning = MagicMock(  # type: ignore[method-assign]
            side_effect=lambda message, extra=None: self.__warnings.append((message, extra))
        )
        self.__logger = SlowQueryLogger(logger=logger, threshold_milliseconds=100, backend="sqlite")

    def test_emit_skipped_when_under_threshold(self) -> None:
        """
        Elapsed durations below the threshold must not emit any warning.
        """

        self.__logger.maybe_emit(sql="SELECT 1", elapsed_milliseconds=50)

        self.assertEqual(0, len(self.__warnings))

    def test_emit_recorded_when_threshold_breached(self) -> None:
        """
        Elapsed durations at or above the threshold must emit one structured warning.
        """

        self.__logger.maybe_emit(sql="SELECT 1", elapsed_milliseconds=250)

        self.assertEqual(1, len(self.__warnings))
        _, extra = self.__warnings[0]
        self.assertEqual(FathomEvent.SLOW_QUERY.value, extra["event"])
        self.assertEqual("sqlite", extra["backend"])
        self.assertEqual(250, extra["elapsed_milliseconds"])

    def test_disabled_threshold_suppresses_emit(self) -> None:
        """
        A non-positive threshold must disable emission entirely.
        """

        disabled = SlowQueryLogger(
            logger=getLogger("fathom.interaction.test.timing.disabled"),
            threshold_milliseconds=0,
            backend="sqlite",
        )

        disabled.maybe_emit(sql="SELECT 1", elapsed_milliseconds=10_000)

        self.assertEqual(0, len(self.__warnings))


class TestTimedConnection(unittest.IsolatedAsyncioTestCase):
    """
    TimedConnection forwards execute() and emits slow-query warnings on threshold breach.
    """

    def setUp(self) -> None:
        """
        Bind a capturing logger that records every warning sent through it.
        """

        self.__warnings: List[tuple] = []
        logger = getLogger("fathom.interaction.test.timing.connection")
        logger.warning = MagicMock(  # type: ignore[method-assign]
            side_effect=lambda message, extra=None: self.__warnings.append((message, extra))
        )
        self.__slow_logger = SlowQueryLogger(
            logger=logger, threshold_milliseconds=50, backend="sqlite"
        )

    async def test_await_path_emits_when_query_is_slow(self) -> None:
        """
        Awaiting a slow execute() resolves and reports a single warning.
        """

        connection = TimedConnection(
            inner=_FakeConnection(delay_seconds=0.1, mode="await"),
            logger=self.__slow_logger,
        )

        result = await connection.execute("UPDATE foo SET bar=1")

        self.assertEqual(1, result.rowcount)
        self.assertEqual(1, len(self.__warnings))

    async def test_async_with_path_emits_when_query_is_slow(self) -> None:
        """
        Entering an async-with execute() over a slow query emits one warning on exit.
        """

        connection = TimedConnection(
            inner=_FakeConnection(delay_seconds=0.1, mode="cursor"),
            logger=self.__slow_logger,
        )

        async with connection.execute("SELECT 1"):
            pass

        self.assertEqual(1, len(self.__warnings))

    async def test_fast_query_emits_no_warning(self) -> None:
        """
        A query that finishes well under the threshold must not emit a warning.
        """

        connection = TimedConnection(
            inner=_FakeConnection(delay_seconds=0.005, mode="cursor"),
            logger=self.__slow_logger,
        )

        async with connection.execute("SELECT 1"):
            pass

        self.assertEqual(0, len(self.__warnings))

    async def test_attribute_delegation_forwards_to_inner(self) -> None:
        """
        Attribute lookups that are not handled by the wrapper must fall through to the inner connection.
        """

        time.sleep(0)
        inner = MagicMock(execute=MagicMock(), commit=MagicMock(return_value="committed"))
        connection = TimedConnection(inner=inner, logger=self.__slow_logger)

        self.assertEqual("committed", connection.commit())
