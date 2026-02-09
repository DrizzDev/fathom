from __future__ import annotations

from typing import Any, Dict, List

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from fathom.schemas.results import ActionResult, PlanResult
from fathom.schemas.screens import ScreenState


class AuditService:
    """
    Service for logging execution details and metrics to the console.
    """

    def __init__(self) -> None:
        self.__console = Console()
        self.__memory_audit_trail: List[Dict[str, Any]] = []

    def log_step(
        self,
        is_stuck: bool,
        step_count: int,
        plan: PlanResult,
        state: ScreenState,
        is_new_screen: bool,
        result: ActionResult,
        total_duration: float,
        analysis_duration: float,
        execution_duration: float,
        grounding_duration: float,
        hierarchy_duration: float,
    ) -> None:
        """
        Prints a detailed audit table for a single execution step.
        """

        audit = Table.grid(padding=(0, 2))
        audit.add_column(style="dim")
        audit.add_column(justify="right")

        status_icon = "🆕" if is_new_screen else "🔄"
        audit.add_row(
            "Screen Status:",
            f"{status_icon} {state.visual_hash[:8]} ({state.activity})",
        )

        if is_stuck:
            audit.add_row("[bold red]Loop Detected:[/bold red]", "YES")

        audit.add_row("Grounding:", self.__format_time(milliseconds=grounding_duration * 1000))

        if hierarchy_duration > 0:
            audit.add_row("Hierarchy:", self.__format_time(milliseconds=hierarchy_duration * 1000))

        if plan.metrics:
            if "memory_retrieval" in plan.metrics:
                audit.add_row(
                    "Memory Retrieval:",
                    self.__format_time(milliseconds=plan.metrics["memory_retrieval"] * 1000),
                )
            if "llm_analysis" in plan.metrics:
                audit.add_row(
                    "LLM Core Analysis:",
                    self.__format_time(milliseconds=plan.metrics["llm_analysis"] * 1000),
                )

        audit.add_row("Total Analysis:", self.__format_time(milliseconds=analysis_duration * 1000))
        audit.add_row("ADB Execution:", self.__format_time(milliseconds=result.duration))

        overhead = (execution_duration * 1000) - result.duration
        audit.add_row("Overhead:", self.__format_time(milliseconds=overhead))

        audit.add_row(
            "[bold white]Total Step Time:[/bold white]",
            f"[bold cyan]{self.__format_time(milliseconds=total_duration * 1000)}[/bold cyan]",
        )

        self.__console.print(
            Panel(
                renderable=audit,
                border_style="dim",
                title_align="right",
                title=f"Step {step_count} Audit",
            )
        )

    def record_context(
        self,
        success: bool,
        step_number: int,
        visual_hash: str,
        context: Dict[str, Any],
        action_description: str,
        knowledge: Dict[str, Any],
    ) -> None:
        """
        Records context data for the final session audit.
        """

        self.__memory_audit_trail.append(
            {
                "step": step_number,
                "context": context,
                "success": success,
                "hash": visual_hash,
                "knowledge": knowledge,
                "action": action_description,
            }
        )

    def print_session_summary(self) -> None:
        """
        Prints the final memory and context audit for the entire session.
        """

        if not self.__memory_audit_trail:
            return

        table = Table(
            show_lines=True,
            header_style="bold magenta",
            title="Execution Context & Memory Audit",
        )

        table.add_column(header="Step", justify="center")
        table.add_column(header="Hash / Knowledge (READ)", style="cyan")
        table.add_column(header="Session Context (SENT)", style="green")
        table.add_column(header="Action Result (WRITE)", style="yellow")

        for item in self.__memory_audit_trail:
            knowledge = item["knowledge"]
            knowledge_string = f"Hash: [dim]{item['hash'][:12]}[/dim]"
            knowledge_string += f"Desc: {knowledge.get('description', 'N/A')}"

            past = knowledge.get("previous_actions", [])
            knowledge_string += (
                f"Past: {len(past)} actions retrieved" if past else "Past: No prior experience"
            )

            context = item["context"]
            failures = context.get("relevant_failures", [])
            context_string = f"History: {context.get('compact_history')}"
            context_string += f"Failures Sent: {', '.join(failures) if failures else 'None'}"

            status = (
                "[bold green]OK[/bold green]" if item["success"] else "[bold red]FAIL[/bold red]"
            )
            action_string = f"{item['action']}\nStatus: {status}"
            table.add_row(str(item["step"]), knowledge_string, context_string, action_string)

        self.__console.print("")
        self.__console.print(table)

    def __format_time(self, milliseconds: float) -> str:
        """
        Formats milliseconds to 'Xs [Yms]' format.
        """

        seconds = milliseconds / 1000.0
        return f"{seconds:.2f}s [{milliseconds:.0f}ms]"
