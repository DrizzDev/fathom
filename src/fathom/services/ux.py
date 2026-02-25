from __future__ import annotations

from typing import Any, Dict, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from fathom.services.text_normalization import normalize_reasoning
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
            "complete_goal": "Goal Status",
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
            grid.add_row("Reasoning:", normalize_reasoning(str(message)))

        if screen_desc := args.get("screen_description", ""):
            grid.add_row("Screen:", screen_desc)

        if action := args.get("action", {}):
            grid.add_row("Action:", action.get("action_type", "?"))
            if action.get("action_type") == "validate":
                if "is_valid" in action:
                    status = (
                        "[bold green]YES[/bold green]"
                        if bool(action.get("is_valid"))
                        else "[bold red]NO[/bold red]"
                    )
                    grid.add_row("Validated:", status)
                if validation_reason := action.get("validation_reason"):
                    grid.add_row("Evidence:", normalize_reasoning(str(validation_reason)))

        if "condition_met" in args:
            condition_met = bool(args.get("condition_met"))
            status = "[bold green]YES[/bold green]" if condition_met else "[bold red]NO[/bold red]"
            grid.add_row("Validated:", status)

        if (evidence := args.get("evidence", "")) and tool_name in ("complete_goal", "verify_goal"):
            grid.add_row("Evidence:", normalize_reasoning(str(evidence)))

        if args.get("goal_completed") or tool_name == "complete_goal":
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
        self.__console.print("")

    def render_fallback(self, reasoning: str, action: str, step_number: int) -> None:
        """
        Renders a simple thinking block for non-tool actions.
        """

        self.__console.print("")
        self.__console.print(
            Panel(
                renderable=(
                    f"[cyan]Reasoning:[/cyan] {normalize_reasoning(reasoning)}\n"
                    f"[yellow]Action:[/yellow] {normalize_reasoning(action)}"
                ),
                title=f"Step {step_number} Thinking",
                border_style="blue",
            )
        )

    def render_hitl_prompt(
        self,
        *,
        step_number: int,
        action: str,
        rationale: str,
        current_intent: Optional[str] = None,
        screen_description: Optional[str] = None,
        decision_keys: str = "a=approve, e=edit intent, r=exit session",
    ) -> None:
        """Render a styled HITL prompt panel."""

        grid = Table.grid(padding=(0, 1))
        grid.add_column(style="cyan", justify="right")
        grid.add_column(style="white")

        grid.add_row("Action:", action)
        grid.add_row("Rationale:", rationale)
        if current_intent:
            grid.add_row("Intent:", current_intent)
        if screen_description:
            grid.add_row("Screen:", screen_description)

        grid.add_row("Decision:", decision_keys)

        self.__console.print("")
        self.__console.print(
            Panel(
                title=f"[bold white]HITL Review[/bold white] [dim]Step {step_number}[/dim]",
                renderable=grid,
                border_style="magenta",
                subtitle_align="right",
                subtitle="[dim]Human-in-the-loop[/dim]",
                expand=False,
            )
        )
