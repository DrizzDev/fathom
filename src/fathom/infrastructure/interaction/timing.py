from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Optional, Protocol, Type

from fathom.constants.events import FathomEvent

if TYPE_CHECKING:
    from logging import Logger
    from types import TracebackType


class _CursorProtocol(Protocol):
    """
    Minimum surface that backend cursors expose to the timing wrapper.
    """

    async def __aenter__(self) -> Any:
        """
        Open the cursor for async-with iteration.
        """

        ...

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        exc_traceback: Optional[TracebackType],
    ) -> None:
        """
        Close the cursor on async-with exit.
        """

        ...


class _ConnectionProtocol(Protocol):
    """
    Minimum surface that backend connections expose to the timing wrapper.
    """

    def execute(self, sql: str, *parameters: Any) -> Any:
        """
        Execute SQL with optional bound parameters.
        """

        ...


class SlowQueryLogger:
    """
    Emit one structured log record per query that exceeds the configured threshold.
    """

    def __init__(
        self,
        *,
        logger: Logger,
        threshold_milliseconds: int,
        backend: str,
    ) -> None:
        """
        Bind the logger, threshold, and backend label used on every emitted record.
        """

        self.__logger = logger
        self.__threshold = threshold_milliseconds
        self.__backend = backend

    @property
    def threshold(self) -> int:
        """
        Expose the configured threshold so callers can short-circuit timing when disabled.
        """

        return self.__threshold

    def maybe_emit(self, *, sql: str, elapsed_milliseconds: float) -> None:
        """
        Emit a slow-query log when the elapsed duration breaches the threshold.
        """

        if self.__threshold <= 0:
            return

        if elapsed_milliseconds < self.__threshold:
            return

        self.__logger.warning(
            "Slow database query observed",
            extra={
                "event": FathomEvent.SLOW_QUERY.value,
                "backend": self.__backend,
                "elapsed_milliseconds": elapsed_milliseconds,
                "threshold_milliseconds": self.__threshold,
                "sql": sql,
            },
        )


class _TimedExecution:
    """
    Wrap one backend execution so its elapsed duration is logged on completion.
    """

    def __init__(
        self,
        *,
        inner: Any,
        sql: str,
        logger: SlowQueryLogger,
    ) -> None:
        """
        Bind the underlying execution, originating SQL, and slow-query logger.
        """

        self.__inner = inner
        self.__sql = sql
        self.__logger = logger
        self.__started: Optional[float] = None

    def __await__(self) -> Any:
        """
        Forward awaited execution while measuring elapsed wall time.
        """

        self.__started = time.perf_counter()

        async def _resolve() -> Any:
            """
            Resolve the inner awaitable and emit a slow-query event on threshold breach.
            """

            try:
                result = await self.__inner
            finally:
                self.__emit_elapsed()
            return result

        return _resolve().__await__()

    async def __aenter__(self) -> Any:
        """
        Open the underlying cursor for async-with iteration while timing it.
        """

        self.__started = time.perf_counter()
        return await self.__inner.__aenter__()

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        exc_traceback: Optional[TracebackType],
    ) -> None:
        """
        Close the underlying cursor and emit a slow-query event on threshold breach.
        """

        try:
            await self.__inner.__aexit__(exc_type, exc_value, exc_traceback)
        finally:
            self.__emit_elapsed()

    def __emit_elapsed(self) -> None:
        """
        Compute and report elapsed duration when timing started.
        """

        if self.__started is None:
            return

        elapsed_milliseconds = (time.perf_counter() - self.__started) * 1000
        self.__logger.maybe_emit(sql=self.__sql, elapsed_milliseconds=elapsed_milliseconds)
        self.__started = None


class TimedConnection:
    """
    Wrap a backend connection so every execute() reports slow queries.
    """

    def __init__(
        self,
        *,
        inner: _ConnectionProtocol,
        logger: SlowQueryLogger,
    ) -> None:
        """
        Bind the underlying connection and slow-query logger.
        """

        self.__inner = inner
        self.__logger = logger

    def execute(self, sql: str, *parameters: Any) -> _TimedExecution:
        """
        Forward an execute() call and time the resulting cursor/awaitable.
        """

        execution = self.__inner.execute(sql, *parameters)
        return _TimedExecution(inner=execution, sql=sql, logger=self.__logger)

    def __getattr__(self, item: str) -> Any:
        """
        Delegate any attribute not handled by the wrapper to the underlying connection.
        """

        return getattr(self.__inner, item)
