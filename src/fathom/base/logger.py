from __future__ import annotations

import sys
from logging import WARNING, FileHandler, Formatter, StreamHandler, getLogger
from pathlib import Path  # noqa: TC003 - runtime use in attach_file_handler
from typing import List, Optional

import structlog

from fathom.settings.env import FathomSettings


class BaseLogger:
    """
    Base logging configuration.
    """

    __configured: bool = False
    __formatter: Optional[Formatter] = None
    __shared_processors: Optional[List[object]] = None

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

        # Shared processors for both structlog and stdlib logging.
        shared_processors = [
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.ExtraAdder(),
            structlog.processors.UnicodeDecoder(),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.stdlib.PositionalArgumentsFormatter(),
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
            renderer = structlog.processors.JSONRenderer()
        else:
            renderer = structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())

        formatter = structlog.stdlib.ProcessorFormatter(
            # These run on stdlib log records that didn't go through structlog
            foreign_pre_chain=shared_processors,
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
        cls.__formatter = formatter
        cls.__shared_processors = shared_processors

    @classmethod
    def attach_file_handler(cls, *, path: Path) -> None:
        """
        Tee the configured structured log stream to ``path`` via a stdlib
        FileHandler that always emits JSON regardless of the console renderer.
        Safe to call after :meth:`configure`; creates parent dirs.
        """

        if not cls.__configured or cls.__shared_processors is None:
            cls.configure()

        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = FileHandler(filename=str(path), encoding="utf-8")
        file_handler.setFormatter(cls.__build_json_formatter())

        getLogger().addHandler(file_handler)

    @classmethod
    def __build_json_formatter(cls) -> Formatter:
        """
        Build a structlog-backed formatter that renders JSON; never adds colors.
        """

        shared_processors = cls.__shared_processors or []
        formatter: Formatter = structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=list(shared_processors),
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                structlog.processors.JSONRenderer(),
            ],
        )
        return formatter
