"""
Opt-in process-level diagnostics installer for runtime hosts.
"""

from __future__ import annotations

import faulthandler
import signal
import threading
from logging import getLogger
from typing import ClassVar

logger = getLogger(__name__)


class RuntimeDiagnostics:
    """
    Register a SIGUSR1 handler that dumps Python thread tracebacks on demand.
    """

    __installed: ClassVar[bool] = False
    __SIGNAL_NAME: ClassVar[str] = "SIGUSR1"

    @classmethod
    def install(cls) -> None:
        """
        Register the signal handler when invoked from the main thread of a POSIX runtime; idempotent and safe.
        """

        if cls.__installed:
            return

        if not hasattr(signal, cls.__SIGNAL_NAME):
            return

        if threading.current_thread() is not threading.main_thread():
            return

        try:
            faulthandler.register(
                chain=True,
                all_threads=True,
                signum=getattr(signal, cls.__SIGNAL_NAME),
            )
        except (ValueError, RuntimeError) as exception:
            logger.warning(
                "could not register traceback dump handler: %s",
                exception,
                extra={"event": "fathom.diagnostics.skipped"},
            )
            return

        cls.__installed = True
        logger.info(
            "registered traceback dump handler",
            extra={
                "signal": cls.__SIGNAL_NAME,
                "event": "fathom.diagnostics.installed",
            },
        )
