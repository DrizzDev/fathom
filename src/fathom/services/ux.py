from __future__ import annotations

from typing import Any, Dict, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from fathom.settings.env import FathomSettings


class UXService:
    """
    Service for handling rich user experience and console output.
    Encapsulates all direct interactions with the Rich library.
    """

    def __init__(self, settings: Optional[FathomSettings] = None) -> None:
        self.__console = Console()
        self.__settings = settings or FathomSettings()

        # Tool categories for visual grouping
        self.__categories = {
            "store_memory": "Memory",
            "recall_memory": "Memory",
            "execute_ui": "UI Actions",
            "verify_goal": "Goal Status",
            "validate_state": "Verification",
        }

    def render_tool_call(self, tool_name: str, args: Dict[str, Any], duration: float = 0.0) -> None:
        """
        Renders a tool call execution block to the console.
        """

        category = self.__categories.get(tool_name, "Operation")

        grid = Table.grid(padding=(0, 1))

        grid.add_column(style="cyan", justify="right")
        grid.add_column(style="white")

        # Assistant Reasoning
        message = args.get("assistant_message", "")
        if message:
            grid.add_row("Reasoning:", message)

        # Actions Summary
        actions = args.get("actions", [])
        if actions:
            types = [item.get("action_type", "?") for item in actions]
            grid.add_row("Actions:", ", ".join(types))

        # Verification Details
        if "condition_met" in args:
            status = (
                "[bold green]YES[/bold green]"
                if args["condition_met"]
                else "[bold red]NO[/bold red]"
            )
            grid.add_row("Validated:", status)

        if "evidence" in args:
            grid.add_row("Evidence:", str(args["evidence"]))

        # Memory Updates
        updates = args.get("memory_updates")
        if updates:
            grid.add_row("Memory:", str(updates))

        # Goal Status
        if args.get("goal_completed"):
            grid.add_row("Status:", "[bold green]GOAL ACHIEVED[/bold green]")

        title = f"[bold white]{tool_name}[/bold white]"
        if duration > 0:
            title += f" [dim]{duration:.2f}s[/dim]"

        self.__console.print(
            Panel(
                expand=False,
                title=title,
                renderable=grid,
                border_style="blue",
                subtitle_align="right",
                subtitle=f"[dim]{category}[/dim]",
            )
        )
