"""UX service for console rendering."""

from __future__ import annotations

from logging import getLogger
from typing import Any, Dict

from rich.console import Console
from rich.panel import Panel

logger = getLogger(__name__)
console = Console()


class UXService:
    """Service for rendering agent state and actions to the console."""

    def render_fallback(
        self,
        reasoning: str,
        action: str,
        step_number: int,
    ) -> None:
        """Render a reasoning and action panel."""
        console.print(f"[bold]Step {step_number}[/bold]")
        console.print(Panel(reasoning, title="🤔 Reasoning", border_style="blue"))
        # Action is usually printed by the execution logic or adapter
