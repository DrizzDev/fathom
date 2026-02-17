from __future__ import annotations

import sys
from logging import WARNING, StreamHandler, getLogger
from typing import Any, Optional, Sequence, cast

import structlog

from fathom.settings.env import FathomSettings


class BaseLogger:
    """
    Base logging configuration.
    """

    __configured: bool = False

    @classmethod
    def configure(cls, settings: Optional[FathomSettings] = None) -> None:
        """
        Configure structured logging based on settings.

        Configures the standard library logging to use structlog formatting.
        This allows standard `logging.getLogger(__name__)` usage to produce structured/JSON output.

        Args:
            settings: FathomSettings instance. If None, uses defaults.
        """

        if cls.__configured:
            return

        if settings is None:
            settings = FathomSettings()

        json_format = settings.log_json
        level_name = settings.log_level.upper()

        # Shared processors for both structlog and stdlib logging
        shared_processors: list[Any] = [
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.UnicodeDecoder(),
        ]

        # Configure structlog
        structlog.configure(
            cache_logger_on_first_use=True,
            wrapper_class=structlog.stdlib.BoundLogger,
            logger_factory=structlog.stdlib.LoggerFactory(),
            processors=[*shared_processors, structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        )

        # Determine renderer
        if json_format:
            renderer: Any = structlog.processors.JSONRenderer()
        else:
            renderer = structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())

        formatter = structlog.stdlib.ProcessorFormatter(
            # These run on stdlib log records that didn't go through structlog
            foreign_pre_chain=cast("Optional[Sequence[Any]]", shared_processors),
            # These run on all log records (structlog or stdlib)
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                renderer,
            ],
        )

        # Configure root logger
        handler = StreamHandler(sys.stderr)
        handler.setFormatter(formatter)

        root_logger = getLogger()
        root_logger.addHandler(handler)
        root_logger.setLevel(level_name)

        # Silence noisy libraries
        for lib in ["httpx", "httpcore", "urllib3", "asyncio", "parso", "aiosqlite"]:
            getLogger(lib).setLevel(WARNING)

        cls.__configured = True
