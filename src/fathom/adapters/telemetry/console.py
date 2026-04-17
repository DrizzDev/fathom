from __future__ import annotations

from typing import Any, Dict, Optional

from rich.console import Console

from fathom.adapters.telemetry.event_panels import render_event_panel
from fathom.constants.events import FathomEvent
from fathom.interfaces.telemetry import TelemetryPort


class ConsoleTelemetryAdapter(TelemetryPort):
    """
    Telemetry adapter that preserves structured logs and adds CLI rendering.
    """

    def __init__(
        self,
        *,
        inner: TelemetryPort,
        console: Optional[Console] = None,
    ) -> None:
        """
        Initialize console telemetry with a wrapped telemetry adapter.
        """

        self.__inner = inner
        self.__console = console or Console()

    async def debug(self, message: str, **context: Any) -> None:
        """
        Publish a debug telemetry event.
        """

        await self.__inner.debug(message, **context)

    async def info(self, message: str, **context: Any) -> None:
        """
        Publish an info telemetry event and render selected CLI events.
        """

        await self.__inner.info(message, **context)
        self.__render(level="info", message=message, context=context)

    async def warning(self, message: str, **context: Any) -> None:
        """
        Publish a warning telemetry event and render CLI warnings.
        """

        await self.__inner.warning(message, **context)
        self.__render(level="warning", message=message, context=context)

    async def error(self, message: str, **context: Any) -> None:
        """
        Publish an error telemetry event and render CLI errors.
        """

        await self.__inner.error(message, **context)
        self.__render(level="error", message=message, context=context)

    async def exception(
        self,
        message: str,
        *,
        exception: Optional[BaseException] = None,
        **context: Any,
    ) -> None:
        """
        Publish an exception telemetry event and render CLI errors.
        """

        await self.__inner.exception(message, exception=exception, **context)
        self.__render(level="error", message=message, context=context)

    def update_identity(self, *, identity: str) -> None:
        """
        Update the wrapped telemetry routing identity when supported.
        """

        if hasattr(self.__inner, "update_identity"):
            self.__inner.update_identity(identity=identity)

    def __render(self, *, level: str, message: str, context: Dict[str, Any]) -> None:
        """
        Render selected telemetry events for CLI operators.

        Delegates panel construction to the shared ``render_event_panel``
        so both the console and the demo TUI produce identical styling.
        Plain-text fallbacks (workflow-lifecycle status lines, bare
        warnings/errors) remain local to this adapter.
        """

        event_type = context.get("type")

        panel = render_event_panel(
            event_type=event_type,
            message=message,
            context=context,
            level=level,
        )
        if panel is not None:
            self.__console.print(panel)
            return

        if event_type in {
            FathomEvent.WORKFLOW_COMPLETED,
            FathomEvent.WORKFLOW_CANCELLED,
            FathomEvent.WORKFLOW_PAUSED,
            FathomEvent.WORKFLOW_RESUMED,
            FathomEvent.HITL_RECEIVED,
        }:
            self.__render_status(message=message, level=level)
            return

        if level == "warning":
            self.__console.print(f"[yellow]{message}[/yellow]")

    def __render_status(self, *, message: str, level: str) -> None:
        """
        Render workflow lifecycle status.
        """

        if level == "error":
            self.__console.print(f"[bold red]{message}[/bold red]")
            return

        self.__console.print(f"[bold blue]{message}[/bold blue]")
