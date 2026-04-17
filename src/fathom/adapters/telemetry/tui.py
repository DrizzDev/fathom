from __future__ import annotations

from typing import Any, Dict, Optional, Protocol

from fathom.interfaces.telemetry import TelemetryPort


class EventSink(Protocol):
    """
    Narrow protocol satisfied by any surface that wants to observe
    telemetry events for UI purposes.

    Decouples :class:`TuiTelemetryAdapter` from the concrete
    ``DemoApp`` — the adapter can notify any sink that exposes
    ``record_event``.
    """

    def record_event(
        self,
        *,
        level: str,
        message: str,
        context: Dict[str, Any],
    ) -> None:
        """
        Receive a telemetry event previously forwarded to the inner
        adapter.
        """

        ...


class TuiTelemetryAdapter(TelemetryPort):
    """
    Forward telemetry to an inner adapter (usually ``StructlogAdapter``)
    and notify an :class:`EventSink` so it can update header/footer
    widgets and push rendered panels into a scrollable body.

    The adapter is task-safe: Textual's ``RichLog.write`` and reactive
    attributes are designed to be set from any coroutine on the same
    event loop; the sink is responsible for marshalling cross-thread
    writes when it runs on a separate thread from the adapter.
    """

    def __init__(self, *, app: EventSink, inner: TelemetryPort) -> None:
        self.__app = app
        self.__inner = inner

    async def debug(self, message: str, **context: Any) -> None:
        await self.__inner.debug(message, **context)

    async def info(self, message: str, **context: Any) -> None:
        await self.__inner.info(message, **context)
        self.__app.record_event(level="info", message=message, context=context)

    async def warning(self, message: str, **context: Any) -> None:
        await self.__inner.warning(message, **context)
        self.__app.record_event(level="warning", message=message, context=context)

    async def error(self, message: str, **context: Any) -> None:
        await self.__inner.error(message, **context)
        self.__app.record_event(level="error", message=message, context=context)

    async def exception(
        self,
        message: str,
        *,
        exception: Optional[BaseException] = None,
        **context: Any,
    ) -> None:
        await self.__inner.exception(message, exception=exception, **context)
        self.__app.record_event(level="error", message=message, context=context)

    def update_identity(self, *, identity: str) -> None:
        """
        Forward identity updates to the wrapped inner telemetry, if it
        supports the optional method.
        """

        if hasattr(self.__inner, "update_identity"):
            self.__inner.update_identity(identity=identity)
