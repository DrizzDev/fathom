from __future__ import annotations

from logging import getLogger

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

console = Console()
logger = getLogger(__name__)


class UXService:
    """
    Service for rendering agent state and actions to the console.
    Provides structured, beautiful output for the CLI.
    """

    def render_step_start(self, step_number: int) -> None:
        """
        Render step header.
        """

        console.rule(f"[bold cyan]Step {step_number}[/bold cyan]")

    def render_reasoning(self, reasoning: str, action_description: str) -> None:
        """
        Render the agent's thought process and planned action.
        """

        # Reasoning Panel
        reasoning_panel = Panel(
            Text(reasoning, style="italic"),
            title="[bold yellow]🤔 Reasoning[/bold yellow]",
            border_style="yellow",
            padding=(1, 2),
        )

        # Action Panel
        action_panel = Panel(
            Text(action_description, style="bold white"),
            title="[bold green]🚀 Planned Action[/bold green]",
            border_style="green",
            padding=(1, 2),
        )

        console.print(reasoning_panel)
        console.print(action_panel)

    def render_fallback(
        self,
        action: str,
        reasoning: str,
        step_number: int,
    ) -> None:
        """
        Render a fallback view when structured rendering is not applicable.
        """

        self.render_step_start(step_number)
        self.render_reasoning(reasoning, action)

    def render_error(self, message: str) -> None:
        """
        Render a prominent error message.
        """

        console.print(
            Panel(
                f"[bold white]{message}[/bold white]",
                title="[bold red]❌ Error[/bold red]",
                border_style="red",
            )
        )
