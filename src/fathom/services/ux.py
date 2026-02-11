from __future__ import annotations

from typing import Any, Dict, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from fathom.settings.env import FathomSettings


class UXService:
    """
    Service for handling rich user experience and console output.
    """

    def __init__(self, settings: Optional[FathomSettings] = None) -> None:
        self.__console = Console()
        self.__settings = settings or FathomSettings()

        self.__categories = {
            "store_memory": "Memory",
            "recall_memory": "Memory",
            "execute_ui": "UI Actions",
            "verify_goal": "Goal Status",
            "validate_state": "Verification",
        }

    def render_tool_call(self, tool_name: str, args: Dict[str, Any], duration: float = 0.0) -> None:
        """
        Renders a structured tool call block.
        """

        category = self.__categories.get(tool_name, "Operation")

        grid = Table.grid(padding=(0, 1))
        grid.add_column(style="cyan", justify="right")
        grid.add_column(style="white")

        if message := args.get("assistant_message", ""):
            grid.add_row("Reasoning:", message)

        if actions := args.get("actions", []):
            types = [item.get("action_type", "?") for item in actions]
            grid.add_row("Actions:", ", ".join(types))

        if condition_met := args.get("condition_met"):
            status = "[bold green]YES[/bold green]" if condition_met else "[bold red]NO[/bold red]"
            grid.add_row("Validated:", status)

        if args.get("goal_completed"):
            grid.add_row("Status:", "[bold green]GOAL ACHIEVED[/bold green]")

        title = f"[bold white]{tool_name}[/bold white] [dim]{duration:.2f}s[/dim]"

        self.__console.print(
            Panel(
                title=title,
                expand=False,
                renderable=grid,
                border_style="blue",
                subtitle_align="right",
                subtitle=f"[dim]{category}[/dim]",
            )
        )

    def render_fallback(self, reasoning: str, action: str, step_number: int) -> None:
        """
        Renders a simple thinking block for non-tool actions.
        """

        self.__console.print(
            Panel(
                renderable=f"[cyan]Reasoning:[/cyan] {reasoning}\n[yellow]Action:[/yellow] {action}",
                title=f"Step {step_number} Thinking",
                border_style="blue",
            )
        )
