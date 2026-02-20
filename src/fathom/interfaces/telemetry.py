from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class TelemetryPort(ABC):
    """
    Abstract interface for telemetry and observability.
    """

    @abstractmethod
    async def debug(self, message: str, **context: Any) -> None:
        """
        Log debug message with context.
        """

        raise NotImplementedError

    @abstractmethod
    async def info(self, message: str, **context: Any) -> None:
        """
        Log info message with context.
        """

        raise NotImplementedError

    @abstractmethod
    async def warning(self, message: str, **context: Any) -> None:
        """
        Log warning message with context.
        """

        raise NotImplementedError

    @abstractmethod
    async def error(self, message: str, **context: Any) -> None:
        """
        Log error message with context.
        """

        raise NotImplementedError
