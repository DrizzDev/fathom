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

        if event_type == FathomEvent.INTENT_CLASSIFIED:
            self.__render_intent_classified(context=context)
            return

        if event_type == FathomEvent.DECOMPOSITION_COMPLETE:
            self.__render_decomposition(context=context)
            return

        if event_type == FathomEvent.SUB_GOAL_STARTED:
            self.__render_sub_goal_started(context=context)
            return

        if event_type == FathomEvent.SUB_GOAL_COMPLETED:
            self.__render_sub_goal_completed(context=context)
            return

        if event_type == FathomEvent.HITL_REQUESTED:
            self.__render_hitl_requested(message=message, context=context)
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

    def __render_intent_classified(self, *, context: Dict[str, Any]) -> None:
        """
        Render the classifier decision as a loud cue panel.
        """

        should_decompose = bool(context.get("should_decompose", True))
        verdict = (
            "[bold magenta]Complex[/bold magenta] task — will be decomposed."
            if should_decompose
            else "[bold green]Simple[/bold green] task — running end-to-end."
        )
        self.__console.print(
            Panel.fit(
                f"[bold #a88fd8]🧠 Understanding your request[/bold #a88fd8]\n{verdict}",
                border_style="#a88fd8",
            )
        )

    def __render_decomposition(self, *, context: Dict[str, Any]) -> None:
        """
        Render the decomposer output as a numbered plan panel.
        """

        raw = context.get("sub_goals") or []
        sub_goals = [str(item) for item in raw if item]
        if not sub_goals:
            return
        numbered = "\n".join(
            f"[bold cyan]{index}[/bold cyan] · {description}"
            for index, description in enumerate(sub_goals, start=1)
        )
        self.__console.print(
            Panel.fit(
                f"[bold #6b3fd4]📋 Plan ({len(sub_goals)} step{'s' if len(sub_goals) != 1 else ''})[/bold #6b3fd4]\n{numbered}",
                border_style="#6b3fd4",
            )
        )

    def __render_sub_goal_started(self, *, context: Dict[str, Any]) -> None:
        """
        Render the start of a sub-goal as a highlighted panel.
        """

        index = context.get("index", 0)
        total = context.get("total", 1)
        description = context.get("description") or "(unnamed)"
        # Human-friendly 1-based display; AgentState indexes from 0.
        human_index = int(index) + 1 if isinstance(index, int) else "?"
        self.__console.print(
            Panel.fit(
                f"[bold cyan]🎯 Starting sub-goal {human_index}/{total}[/bold cyan]\n{description}",
                border_style="cyan",
            )
        )

    def __render_sub_goal_completed(self, *, context: Dict[str, Any]) -> None:
        """
        Render sub-goal completion as a green check panel.
        """

        index = context.get("index", 0)
        total = context.get("total", 1)
        description = context.get("description") or "(unnamed)"
        human_index = int(index) + 1 if isinstance(index, int) else "?"
        self.__console.print(
            Panel.fit(
                f"[bold green]✓ Completed sub-goal {human_index}/{total}[/bold green]\n{description}",
                border_style="green",
            )
        )

    def __render_hitl_requested(self, *, message: str, context: Dict[str, Any]) -> None:
        """
        Render a loud HITL pause panel so the handoff is obvious.
        """

        prompt = context.get("prompt") or message
        self.__console.print(
            Panel.fit(
                f"[bold yellow]⏸  Awaiting your input[/bold yellow]\n{prompt}",
                border_style="yellow",
            )
        )
