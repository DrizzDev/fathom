from __future__ import annotations

from typing import Any, Dict, Optional

from rich.console import Console
from rich.panel import Panel

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
        Wrap an inner telemetry adapter and bind a Rich console for CLI rendering.
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
        """

        event_type = context.get("type")

        if event_type == FathomEvent.REASONING:
            self.__render_reasoning(message=message, context=context)
            return

        if event_type == FathomEvent.PLANNED_ACTION:
            self.__render_planned_action(message=message, context=context)
            return

        if event_type == FathomEvent.STEP_COMPLETED:
            self.__render_step_completed(context=context)
            return

        if event_type in {
            FathomEvent.WORKFLOW_COMPLETED,
            FathomEvent.WORKFLOW_CANCELLED,
            FathomEvent.WORKFLOW_PAUSED,
            FathomEvent.WORKFLOW_RESUMED,
            FathomEvent.HITL_REQUESTED,
            FathomEvent.HITL_RECEIVED,
        }:
            self.__render_status(message=message, level=level)
            return

        if level == "warning":
            self.__console.print(f"[yellow]{message}[/yellow]")
            return

        if level == "error":
            self.__console.print(f"[bold red]{message}[/bold red]")

    def __render_reasoning(self, *, message: str, context: Dict[str, Any]) -> None:
        """
        Render reasoning details for the current step.
        """

        step_number = context.get("step", "?")
        reasoning = context.get("reasoning") or message
        self.__console.print(
            Panel.fit(
                f"[bold cyan]Step {step_number} Reasoning[/bold cyan]\n{reasoning}",
                border_style="cyan",
            )
        )

    def __render_planned_action(self, *, message: str, context: Dict[str, Any]) -> None:
        """
        Render the planned action for the current step.
        """

        step_number = context.get("step", "?")
        self.__console.print(
            Panel.fit(
                f"[bold green]Step {step_number} Action[/bold green]\n{message}",
                border_style="green",
            )
        )

    def __render_step_completed(self, *, context: Dict[str, Any]) -> None:
        """
        Render step completion summary.
        """

        step_number = context.get("step", "?")
        success = bool(context.get("success"))
        action_description = context.get("action_description") or "unknown"
        observation = context.get("observation") or "N/A"
        border_style = "green" if success else "red"
        title = "Success" if success else "Failed"
        self.__console.print(
            Panel.fit(
                (
                    f"[bold]{title}[/bold] | Step {step_number}\n"
                    f"Action: {action_description}\n"
                    f"Observation: {observation}"
                ),
                border_style=border_style,
            )
        )

    def __render_status(self, *, message: str, level: str) -> None:
        """
        Render workflow lifecycle status.
        """

        if level == "error":
            self.__console.print(f"[bold red]{message}[/bold red]")
            return

        self.__console.print(f"[bold blue]{message}[/bold blue]")
