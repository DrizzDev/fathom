"""Fathom structured logging configuration."""

from __future__ import annotations

import logging
import sys
from typing import Any, Optional

import structlog
from structlog.types import Processor


def configure_logging(
    *,
    level: int = logging.INFO,
    json_format: bool = False,
    add_timestamp: bool = True,
) -> None:
    """Configure structured logging for Fathom.

    Args:
        level: Logging level (e.g., logging.DEBUG, logging.INFO).
        json_format: If True, output JSON logs. Otherwise, human-readable.
        add_timestamp: If True, add ISO timestamp to log entries.
    """
    processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if add_timestamp:
        processors.insert(0, structlog.processors.TimeStamper(fmt="iso"))

    if json_format:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(
            structlog.dev.ConsoleRenderer(
                colors=sys.stderr.isatty(),
            )
        )

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stderr,
        level=level,
    )


def get_logger(name: str, **initial_context: Any) -> structlog.stdlib.BoundLogger:
    """Get a structured logger with optional initial context.

    Args:
        name: Logger name (typically __name__).
        **initial_context: Key-value pairs to bind to all log entries.

    Returns:
        Configured structured logger.
    """
    logger: structlog.stdlib.BoundLogger = structlog.get_logger(name)
    if initial_context:
        logger = logger.bind(**initial_context)
    return logger


class LogContext:
    """Context manager for scoped log context."""

    def __init__(self, **context: Any) -> None:
        self.__context = context
        self.__token: Optional[object] = None

    def __enter__(self) -> "LogContext":
        self.__token = structlog.contextvars.bind_contextvars(**self.__context)
        return self

    def __exit__(self, *_: Any) -> None:
        if self.__token is not None:
            structlog.contextvars.unbind_contextvars(*self.__context.keys())

    def bind(self, **extra: Any) -> None:
        """Add additional context within this scope."""
        structlog.contextvars.bind_contextvars(**extra)
