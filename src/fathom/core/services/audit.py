from __future__ import annotations

from typing import Any, Dict, List, Union

from rich.console import Console
from rich.markup import escape
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
            f"{status_icon} {state.visual_hash[:12]} ({state.activity})",
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

            # Token usage
            prompt_tokens = int(plan.metrics.get("prompt_tokens", 0))
            completion_tokens = int(plan.metrics.get("completion_tokens", 0))
            cached_tokens = int(plan.metrics.get("cached_tokens", 0))
            if prompt_tokens or completion_tokens:
                total_tokens = prompt_tokens + completion_tokens
                token_str = f"{total_tokens:,} (prompt: {prompt_tokens:,} | completion: {completion_tokens:,}"
                if cached_tokens:
                    token_str += f" | cached: {cached_tokens:,}"
                token_str += ")"
                audit.add_row("Tokens:", f"[dim]{token_str}[/dim]")

        audit.add_row("Total Analysis:", self.__format_time(milliseconds=analysis_duration * 1000))
        audit.add_row("ADB Execution:", self.__format_time(milliseconds=result.duration))

        overhead = (execution_duration * 1000) - result.duration
        audit.add_row("Overhead:", self.__format_time(milliseconds=overhead))

        audit.add_row(
            "[bold white]Total Step Time:[/bold white]",
            f"[bold cyan]{self.__format_time(milliseconds=total_duration * 1000)}[/bold cyan]",
        )

        # Cache Stats
        if plan.metrics and "cache_hits" in plan.metrics:
            hits = int(plan.metrics.get("cache_hits", 0))
            misses = int(plan.metrics.get("cache_misses", 0))
            audit.add_row("Cache:", f"hits: {hits} | misses: {misses}")

        self.__console.print(
            Panel(
                renderable=audit,
                border_style="dim",
                title_align="right",
                title=f"Step {step_count} Audit",
            )
        )
        
        # --- PROMPT AUDIT ---
        prompt_payload = plan.metadata.get("prompt_payload")
        if prompt_payload:
            sanitized = self.__sanitize_for_log(prompt_payload)
            # Sanitized is a list (because payload is list).
            # Convert to string, escaping markup characters in the content
            prompt_string = "\n".join(
                escape(str(item)) for item in sanitized
            )
            
            if "USER INSTRUCTION" in prompt_string:
                prompt_string = prompt_string.replace(
                    "USER INSTRUCTION (PRIORITY)", 
                    "[bold red]USER INSTRUCTION (PRIORITY)[/bold red]"
                )
            
            self.__console.print(
                Panel(
                    prompt_string,
                    title="[bold blue]Context & Prompt[/bold blue]",
                    border_style="blue",
                    expand=False
                )
            )

        # Print Reasoning
        reasoning = plan.reason

        if plan.step and plan.step.action:
            reasoning = plan.step.action.rationale or reasoning

            # Show Action Details
            action_panel = Table.grid(padding=(0, 2))
            action_panel.add_column(style="bold yellow")
            action_panel.add_column()

            action_panel.add_row("Action:", plan.step.action.action_type.value)
            action_panel.add_row("Target:", plan.step.action.target or "N/A")
            action_panel.add_row("Rationale:", reasoning)

            self.__console.print(
                Panel(action_panel, title="LLM Reasoning & Action", border_style="yellow")
            )
        elif reasoning:
            self.__console.print(
                Panel(f"[italic]{reasoning}[/italic]", title="LLM Reasoning", border_style="yellow")
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
                "context": self.__sanitize_for_log(context),
                "success": success,
                "hash": visual_hash,
                "knowledge": self.__sanitize_for_log(knowledge),
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
            context_string += f"Failures Sent: {', '.join(str(f) for f in failures) if failures else 'None'}"

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

    def __sanitize_for_log(self, data: Any) -> Any:
        """
        Recursively sanitizes data structure to remove bytes and large objects.
        """
        if isinstance(data, bytes):
            return f"<bytes len={len(data)}>"
        if isinstance(data, dict):
            return {k: self.__sanitize_for_log(v) for k, v in data.items()}
        if isinstance(data, list):
            return [self.__sanitize_for_log(v) for v in data]
        if isinstance(data, tuple):
            return tuple(self.__sanitize_for_log(v) for v in data)
        return data
