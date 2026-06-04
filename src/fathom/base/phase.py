"""
Bounded and abandonable phase primitives for instrumented finalization awaits.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from logging import INFO, WARNING, Logger, getLogger
from typing import Any, Awaitable, Dict, Generic, Optional, TypeVar

from fathom.constants.finalization import FinalizationPhase
from fathom.core.exceptions import FinalizationTimeoutError

_RESULT = TypeVar("_RESULT")

logger = getLogger(__name__)


class _PhaseLogger:
    """
    Internal helper that emits structured phase boundary log records with correlation fields.
    """

    def __init__(
        self, *, phase: FinalizationPhase, timeout: float, workflow_id: Optional[str]
    ) -> None:
        """
        Bind phase identity and correlation fields for downstream log emission.
        """

        self.__phase = phase
        self.__timeout = timeout
        self.__workflow_id = workflow_id

    def emit(
        self,
        *,
        level: int,
        suffix: str,
        sink: Logger = logger,
        duration: Optional[float] = None,
        exception: Optional[BaseException] = None,
    ) -> None:
        """
        Emit one structured log record for a phase boundary transition.
        """

        event = f"{self.__phase.value}.{suffix}"

        extra: Dict[str, Any] = {
            "event": event,
            "timeout": self.__timeout,
            "phase": self.__phase.value,
        }
        if self.__workflow_id is not None:
            extra["workflow.id"] = self.__workflow_id

        if duration is not None:
            extra["duration"] = duration

        if exception is not None:
            extra["exception.message"] = str(exception)
            extra["exception.type"] = type(exception).__name__

        sink.log(level, "phase=%s event=%s", self.__phase.value, event, extra=extra)


class BoundedPhase(Generic[_RESULT]):
    """
    Run an awaitable that must complete within a deadline; raise FinalizationTimeoutError on overrun.
    """

    def __init__(
        self,
        *,
        timeout: float,
        phase: FinalizationPhase,
        workflow_id: Optional[str] = None,
    ) -> None:
        """
        Bind phase identity and timeout to this bounded boundary.
        """

        self.__phase = phase
        self.__timeout = timeout
        self.__workflow_id = workflow_id
        self.__log = _PhaseLogger(phase=phase, timeout=timeout, workflow_id=workflow_id)

    async def execute(self, *, awaitable: Awaitable[_RESULT]) -> _RESULT:
        """
        Await the inner awaitable under the configured deadline.
        """

        started_at = time.perf_counter()
        self.__log.emit(suffix="started", level=INFO)

        try:
            result = await asyncio.wait_for(awaitable, timeout=self.__timeout)
        except asyncio.TimeoutError as exception:
            self.__log.emit(
                level=WARNING,
                suffix="timed_out",
                duration=time.perf_counter() - started_at,
            )
            raise FinalizationTimeoutError(
                timeout=self.__timeout,
                phase=self.__phase.value,
                workflow_id=self.__workflow_id,
            ) from exception
        except asyncio.CancelledError:
            raise
        except Exception as exception:
            self.__log.emit(
                level=WARNING,
                suffix="failed",
                exception=exception,
                duration=time.perf_counter() - started_at,
            )
            raise

        self.__log.emit(
            level=INFO,
            suffix="completed",
            duration=time.perf_counter() - started_at,
        )
        return result


class AbandonablePhase:
    """
    Run an awaitable that must not gate caller return; on timeout cancel and abandon without awaiting cancellation.
    """

    def __init__(
        self,
        *,
        phase: FinalizationPhase,
        timeout: float,
        workflow_id: Optional[str] = None,
    ) -> None:
        """
        Bind phase identity and timeout to this abandonable boundary.
        """

        self.__phase = phase
        self.__timeout = timeout
        self.__workflow_id = workflow_id
        self.__log = _PhaseLogger(phase=phase, timeout=timeout, workflow_id=workflow_id)

    async def execute(self, *, awaitable: Awaitable[Any]) -> Optional[Any]:
        """
        Run the awaitable as a task; return its result if it finishes within the deadline, else abandon and return None.
        """

        self.__log.emit(suffix="started", level=INFO)
        started_at = time.perf_counter()
        task: asyncio.Task[Any] = asyncio.create_task(
            self.__coerce(awaitable=awaitable),
            name=self.__phase.value,
        )
        try:
            done, pending = await asyncio.wait({task}, timeout=self.__timeout)
        except asyncio.CancelledError:
            task.cancel()
            raise
        if task in pending:
            task.cancel()
            task.add_done_callback(self.__on_abandoned_settled)
            self.__log.emit(
                suffix="abandoned",
                level=WARNING,
                duration=time.perf_counter() - started_at,
            )
            return None
        try:
            result = task.result()
        except asyncio.CancelledError:
            self.__log.emit(
                suffix="abandoned",
                level=WARNING,
                duration=time.perf_counter() - started_at,
            )
            return None
        except Exception as exception:
            self.__log.emit(
                suffix="failed",
                level=WARNING,
                duration=time.perf_counter() - started_at,
                exception=exception,
            )
            return None
        self.__log.emit(
            suffix="completed",
            level=INFO,
            duration=time.perf_counter() - started_at,
        )
        return result

    @staticmethod
    async def __coerce(*, awaitable: Awaitable[Any]) -> Any:
        """
        Adapt an arbitrary awaitable so it can be wrapped in create_task with a stable signature.
        """

        return await awaitable

    def __on_abandoned_settled(self, task: asyncio.Task[Any]) -> None:
        """
        Log the eventual outcome of an abandoned task; never propagate exceptions.
        """

        try:
            if task.cancelled():
                self.__log.emit(suffix="cancel_settled", level=WARNING)
                return
            exception = task.exception()
            if exception is not None:
                self.__log.emit(suffix="settled_with_error", level=WARNING, exception=exception)
                return
            self.__log.emit(suffix="settled_after_abandon", level=INFO)
        except BaseException as callback_exception:
            with contextlib.suppress(BaseException):
                logger.error(
                    "phase=%s event=callback_error",
                    self.__phase.value,
                    extra={
                        "event": f"{self.__phase.value}.callback_error",
                        "phase": self.__phase.value,
                        "exception.type": type(callback_exception).__name__,
                        "exception.message": str(callback_exception),
                    },
                )
