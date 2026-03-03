from __future__ import annotations

from logging import getLogger
from typing import Any

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

    async def debug(self, message: str, **context: Any) -> None:
        """
        Log debug message with context.
        """

        self.__logger.debug(message, extra=context)

    async def info(self, message: str, **context: Any) -> None:
        """
        Log info message with context.
        """

        self.__logger.info(message, extra=context)

    async def warning(self, message: str, **context: Any) -> None:
        """
        Log warning message with context.
        """

        self.__logger.warning(message, extra=context)

    async def error(self, message: str, **context: Any) -> None:
        """
        Log error message with context.
        """

        self.__logger.error(message, extra=context)
