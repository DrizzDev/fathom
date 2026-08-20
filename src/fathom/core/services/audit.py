from __future__ import annotations

import time
from collections import deque
from typing import Any, Dict, List, Optional

from fathom.core.context.manager import ContextManager
from fathom.schemas.results import PlanResult
from fathom.schemas.screens import ScreenState
from fathom.schemas.steps import StepRecord


class AuditService:
    """
    Renders per-step execution audits, prompts, and context to a Rich console.
    """

    def __init__(self, *, console: Optional[Any] = None) -> None:
        """
        Initialize auditor with injected or default console.
        """

        self.__console = console or self.__build_console()
        self.__memory_audit_trail: deque[Dict[str, Any]] = deque(maxlen=1000)

    def log_context(self, manager: ContextManager) -> None:
        """
        Visualizes the current three-tier execution context.
        """

        full_context = manager.get_full_context()
        Panel, Table, escape = self.__rich_types()

        roadmap = full_context.get("roadmap")
        roadmap_table = Table.grid(padding=(0, 1))
        roadmap_table.add_column(style="bold blue")
        roadmap_table.add_column()

        intent = getattr(roadmap, "intent", "Unknown")
        roadmap_table.add_row("Intent:", escape(str(intent)))

        if guidance := manager.get_user_guidance():
            guidance_panel = Table.grid(padding=(0, 1))
            for instruction in guidance:
                time_str = time.strftime("%H:%M:%S", time.localtime(instruction.timestamp))
                guidance_panel.add_row(
                    f"[dim][{time_str}][/dim]",
                    f"[bold red]![/bold red] {escape(instruction.content)}",
                )

            self.__console.print(
                Panel(
                    guidance_panel,
                    title="[bold red]User Instructions (Priority)[/bold red]",
                    border_style="red",
                )
            )

        summary = Table.grid(padding=(0, 1))
        summary.add_column(style="dim")
        summary.add_column()

        if milestones := full_context.get("milestones", []):
            summary.add_row(
                "Milestones:", ", ".join(escape(str(milestone)) for milestone in milestones)
            )

        trace = full_context.get("trace", [])
        summary.add_row("Trace Size:", f"{len(trace)} cycles recorded")

        self.__console.print(
            Panel(
                summary,
                title="[bold blue]Execution Context (GCC-Inspired)[/bold blue]",
                border_style="blue",
            )
        )

    def log_prompt(self, payload: List[Any], instruction: str) -> None:
        """
        Visualizes the exact prompt being sent to the LLM.
        """

        sanitized_payload = self.__sanitize_recursive(data=payload)
        Panel, Table, escape = self.__rich_types()

        prompt_table = Table.grid(padding=(0, 1))
        prompt_table.add_column(style="cyan", no_wrap=True)
        prompt_table.add_column()

        for index, item in enumerate(sanitized_payload):
            content = str(item)
            label = f"Part {index + 1}:"

            # Escape first, then inject Rich markup so tags aren't escaped
            content = escape(content)
            if "USER INSTRUCTIONS" in content:
                content = content.replace(
                    "USER INSTRUCTIONS", "[bold red]USER INSTRUCTIONS[/bold red]"
                )

            prompt_table.add_row(label, content)

        self.__console.print(
            Panel(
                prompt_table,
                title="[bold cyan]🚀 LLM Dispatch[/bold cyan]",
                subtitle=f"[dim]System: {escape(instruction[:80])}...[/dim]",
                border_style="cyan",
            )
        )

    def log_step(
        self,
        plan: PlanResult,
        state: ScreenState,
        result: StepRecord,
        *,
        step_count: int,
        is_new_screen: bool,
        is_stuck: bool,
        total_duration: float,
        analysis_duration: float,
        grounding_duration: float,
        hierarchy_duration: float,
        execution_duration: float,
    ) -> None:
        """
        Prints the detailed audit for a single execution step.
        """

        _ = analysis_duration
        _ = execution_duration
        Panel, Table, escape = self.__rich_types()

        audit_grid = Table.grid(padding=(0, 2))

        audit_grid.add_column(style="dim")
        audit_grid.add_column(justify="right")

        status_tag = "🆕" if is_new_screen else "🔄"
        audit_grid.add_row(
            "Screen:",
            f"{status_tag} {state.visual_hash[:12]} ({state.activity})",
        )

        if is_stuck:
            audit_grid.add_row("[bold red]STUCK:[/bold red]", "YES")

        audit_grid.add_row("Grounding:", self.__format_ms(seconds=grounding_duration))

        if hierarchy_duration > 0:
            audit_grid.add_row("Hierarchy:", self.__format_ms(seconds=hierarchy_duration))

        if plan.metrics:
            if "llm_analysis" in plan.metrics:
                audit_grid.add_row(
                    "LLM Analysis:", self.__format_ms(seconds=plan.metrics["llm_analysis"])
                )

            prompt_t = int(plan.metrics.get("prompt_tokens", 0))
            completion_t = int(plan.metrics.get("completion_tokens", 0))

            if prompt_t or completion_t:
                token_info = f"{prompt_t + completion_t:,} (P:{prompt_t:,} | C:{completion_t:,})"
                audit_grid.add_row("Tokens:", f"[dim]{token_info}[/dim]")

        if plan.step and plan.step.action:
            confidence_pct = plan.step.action.confidence * 100
            confidence_color = (
                "green"
                if plan.step.action.confidence >= 0.7
                else "yellow"
                if plan.step.action.confidence >= 0.4
                else "red"
            )
            audit_grid.add_row(
                "Confidence:", f"[{confidence_color}]{confidence_pct:.1f}%[/{confidence_color}]"
            )

        audit_grid.add_row("Device Command:", self.__format_ms(milliseconds=float(result.duration)))

        audit_grid.add_row(
            "[bold white]Total Time:[/bold white]",
            f"[bold cyan]{self.__format_ms(seconds=total_duration)}[/bold cyan]",
        )

        self.__console.print(
            Panel(
                audit_grid,
                title=f"Step {step_count} Result",
                border_style="dim",
                title_align="right",
            )
        )

        action_info = Table.grid(padding=(0, 2))
        action_info.add_column(style="bold yellow")
        action_info.add_column()

        action_info.add_row("Action:", escape(str(result.action_description or result.action_type)))
        action_info.add_row("Target:", escape(str(result.natural_language_target or result.target)))

        if result.observation:
            action_info.add_row("Observation:", escape(str(result.observation)))

        action_info.add_row("Rationale:", escape(str(result.rationale or "N/A")))

        self.__console.print(Panel(action_info, title="Brain Reasoning", border_style="yellow"))

    def __format_ms(self, seconds: float = 0, milliseconds: float = 0) -> str:
        """
        Format a duration as seconds with a bracketed millisecond value.
        """

        total_ms = (seconds * 1000) + milliseconds
        return f"{total_ms / 1000.0:.2f}s [{total_ms:.0f}ms]"

    def __sanitize_recursive(self, data: Any) -> Any:
        """
        Recursively replaces bytes and large objects for logging.
        """

        if isinstance(data, bytes):
            return f"<binary data: {len(data)} bytes>"

        if isinstance(data, dict):
            return {key: self.__sanitize_recursive(data=value) for key, value in data.items()}

        if isinstance(data, (list, tuple)):
            return [self.__sanitize_recursive(data=item) for item in data]

        return data

    @staticmethod
    def __build_console() -> Any:
        """
        Build the Rich console dependency for audit rendering.
        """

        try:
            from rich.console import Console
        except ModuleNotFoundError as exception:
            raise RuntimeError("Rich is required to instantiate AuditService.") from exception
        return Console()

    @staticmethod
    def __rich_types() -> tuple[Any, Any, Any]:
        """
        Resolve Rich rendering helpers lazily.
        """

        try:
            from rich.markup import escape
            from rich.panel import Panel
            from rich.table import Table
        except ModuleNotFoundError as exception:
            raise RuntimeError("Rich is required to use AuditService rendering.") from exception
        return Panel, Table, escape
