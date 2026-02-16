"""Audit service for logging and session summaries."""

from __future__ import annotations

from logging import getLogger

from rich.console import Console
from rich.table import Table

from fathom.schemas.results import ActionResult, PlanResult
from fathom.schemas.screens import ScreenState

logger = getLogger(__name__)
console = Console()


class AuditService:
    """Service for logging steps and printing session summaries."""

    def log_step(
        self,
        plan: PlanResult,
        state: ScreenState,
        result: ActionResult,
        is_new_screen: bool,
        is_stuck: bool,
        step_count: int,
        analysis_duration: float,
        grounding_duration: float,
        hierarchy_duration: float,
        execution_duration: float,
        total_duration: float,
    ) -> None:
        """Log a single execution step."""
        logger.info(
            f"Step {step_count} completed in {total_duration:.2f}s. "
            f"Success: {result.success}, New Screen: {is_new_screen}, Stuck: {is_stuck}"
        )

    def print_session_summary(self) -> None:
        """Print a summary of the execution session."""
        table = Table(title="Execution Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        # Logic to be populated from session data
        console.print(table)
