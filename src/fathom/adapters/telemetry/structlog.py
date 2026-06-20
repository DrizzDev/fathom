from __future__ import annotations

from logging import getLogger
from typing import Any, Optional

from fathom.interfaces.telemetry import TelemetryPort


class StructlogAdapter(TelemetryPort):
    """
    Structlog adapter for telemetry.
    Uses Python's standard logging with structlog for structured logging.
    """

    def __init__(self, *, logger_name: str = "fathom") -> None:
        """
        Initialize structlog adapter.
        """

        self.__logger = getLogger(name=logger_name)

    async def debug(self, text: str, **context: Any) -> None:
        """
        Log debug message with context.
        """

        self.__logger.debug(text, extra=context)

    async def info(self, text: str, **context: Any) -> None:
        """
        Log info message with context.
        """

        self.__logger.info(text, extra=context)

    async def warning(self, text: str, **context: Any) -> None:
        """
        Log warning message with context.
        """

        self.__logger.warning(text, extra=context)

    async def error(self, text: str, **context: Any) -> None:
        """
        Log error message with context.
        """

        self.__logger.error(text, extra=context)

    async def exception(
        self,
        text: str,
        *,
        exception: Optional[BaseException] = None,
        **context: Any,
    ) -> None:
        """
        Log error message together with exception context.
        """

        if exception is None:
            self.__logger.exception(text, extra=context)
            return

        self.__logger.error(
            text,
            extra=context,
            exc_info=(type(exception), exception, exception.__traceback__),
        )
