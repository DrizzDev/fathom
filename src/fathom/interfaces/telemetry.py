from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal, Optional, TypeAlias

TelemetryLevel: TypeAlias = Literal["debug", "info", "warning", "error"]


class TelemetryPort(ABC):
    """
    Abstract interface for telemetry and observability.
    """

    @abstractmethod
    async def debug(self, text: str, **context: Any) -> None:
        """
        Log debug message with context.
        """

        raise NotImplementedError

    @abstractmethod
    async def info(self, text: str, **context: Any) -> None:
        """
        Log info message with context.
        """

        raise NotImplementedError

    @abstractmethod
    async def warning(self, text: str, **context: Any) -> None:
        """
        Log warning message with context.
        """

        raise NotImplementedError

    @abstractmethod
    async def error(self, text: str, **context: Any) -> None:
        """
        Log error message with context.
        """

        raise NotImplementedError

    @abstractmethod
    async def exception(
        self,
        text: str,
        *,
        exception: Optional[BaseException] = None,
        **context: Any,
    ) -> None:
        """
        Log error message with exception context.
        """

        raise NotImplementedError
