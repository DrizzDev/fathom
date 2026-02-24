import argparse
import asyncio
import contextlib
import os
import signal
import sys
from logging import getLogger
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from fathom.schemas.results import ExplorationResult

os.environ.setdefault("GRPC_VERBOSITY", "ERROR")

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from fathom.base.logger import BaseLogger
from fathom.exceptions import FathomError
from fathom.orchestration.runner import FathomRunner
from fathom.settings.env import FathomSettings
from fathom.utils.cli_input import poll_input_line

console = Console()
logger = getLogger(__name__)


class FathomCLI:
    """
    Fathom CLI application
    """

    def __init__(self, settings: FathomSettings) -> None:
        """
        Initialize CLI with settings.
        """

        self.settings = settings
        self.runner = FathomRunner(settings)

    def __setup_signals(self) -> None:
        """
        Configure signal handlers for graceful shutdown.
        Must be called within an active event loop.
        """

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop = asyncio.get_running_loop()
                loop.add_signal_handler(sig, self.__handle_interrupt)
            except (NotImplementedError, ValueError, RuntimeError):
                pass

        pause_sig = getattr(signal, "SIGUSR1", None)
        if pause_sig:
            try:
                loop = asyncio.get_running_loop()
                loop.add_signal_handler(pause_sig, self.__handle_pause_toggle)
            except (NotImplementedError, ValueError, RuntimeError):
                pass

    def __handle_interrupt(self) -> None:
        """
        Handler called when a signal is received.
        """

        console.print("\n[bold yellow]Stopping gracefully... Please wait.[/bold yellow]")
        self.runner.cancel()

    def __handle_pause_toggle(self) -> None:
        """Toggle pause/resume for the active workflow."""

        if self.runner.is_paused():
            self.runner.resume()
            console.print("\n[bold green]Resumed.[/bold green]")
        else:
            self.runner.pause()
            console.print(
                "\n[bold yellow]Paused. Type 'p' + Enter or send SIGUSR1 to resume.[/bold yellow]"
            )

    async def __listen_for_pause(self) -> None:
        """Listen for pause/resume commands on stdin (p + Enter)."""

        while True:
            line = poll_input_line()
            if line:
                command = line.strip().lower()
                if (
                    command in ("p", "pause")
                    or command in ("resume", "r")
                    and self.runner.is_paused()
                ):
                    self.__handle_pause_toggle()
            await asyncio.sleep(0.2)

    async def run(
        self,
        intent: str,
        max_steps: int = 100,
        device_serial: Optional[str] = None,
        human_in_loop: bool = False,
        **kwargs: Any,
    ) -> int:
        """
        Run an intent workflow with rich UI.
        """

        self.__setup_signals()

        console.print(
            Panel.fit(
                f"[bold blue]Fathom Agent[/bold blue]\n[cyan]Intent:[/cyan] {intent}",
                border_style="blue",
            )
        )
        console.print(
            "[dim]Pause/Resume: type 'p' + Enter or send SIGUSR1. HITL prompts show only while paused.[/dim]"
        )

        pause_task: Optional[asyncio.Task[None]] = None
        try:
            pause_task = asyncio.create_task(self.__listen_for_pause())
            console.print("[bold green]Agent working...[/bold green]")
            result = await self.runner.run_intent(
                intent=intent,
                max_steps=max_steps,
                device_serial=device_serial,
                use_xml=kwargs.get("use_xml", False),
                prompt_version=kwargs.get("prompt_version"),
                human_in_loop=human_in_loop,
            )
        except (asyncio.CancelledError, KeyboardInterrupt):
            console.print("\n[bold red]Execution cancelled by user.[/bold red]")
            return 1
        except FathomError as exception:
            logger.error(f"CLI Error: {exception}")
            console.print(f"[bold red]Fathom Error:[/bold red] {exception}")
            return 1
        except Exception as exception:
            logger.exception("Unexpected error")
            console.print(f"[bold red]Unexpected Error:[/bold red] {exception}")
            return 1
        finally:
            if pause_task:
                pause_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await pause_task

        # Execution Summary
        table = Table(title="Execution Summary", border_style="green" if result.success else "red")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")

        table.add_row("Status", "[green]Success[/green]" if result.success else "[red]Failed[/red]")
        table.add_row("Reason", result.completion_reason)
        table.add_row("Steps Taken", str(result.steps_taken))
        console.print(table)

        # Timing Audit
        if result.metrics:
            audit_table = Table(title="Timing Audit", border_style="blue")
            audit_table.add_column("Operation", style="cyan")
            audit_table.add_column("Total Time (s)", style="magenta", justify="right")
            audit_table.add_column("Avg/Step (s)", style="yellow", justify="right")

            token_metrics = result.metrics.get("Tokens")

            for operation, data in result.metrics.items():
                if operation == "Tokens":
                    continue

                total = data.get("total", 0.0)
                avg = data.get("avg", 0.0)
                audit_table.add_row(operation, f"{total:.2f}s", f"{avg:.2f}s")

            console.print(audit_table)

            if token_metrics:
                token_table = Table(title="Resource Usage (Tokens)", border_style="yellow")
                token_table.add_column("Metric", style="cyan")
                token_table.add_column("Value", style="magenta", justify="right")

                token_table.add_row("Prompt Tokens", f"{token_metrics.get('prompt', 0):,.0f}")
                token_table.add_row(
                    "Completion Tokens", f"{token_metrics.get('completion', 0):,.0f}"
                )
                token_table.add_row("Cached Tokens", f"{token_metrics.get('cached', 0):,.0f}")
                token_table.add_row("Total Tokens", f"{token_metrics.get('total', 0):,.0f}")

                console.print(token_table)

        # Memory / Knowledge Graph Summary
        if result.memory_summary:
            memory_table = Table(title="Agent Knowledge Graph (Brain)", border_style="cyan")
            memory_table.add_column("Screen Hash", style="dim")
            memory_table.add_column("Activity")
            memory_table.add_column("Agent Description", style="italic")

            screens = result.memory_summary.get("screens", [])
            for screen in screens[:10]:  # Show last 10
                memory_table.add_row(
                    screen["hash"], screen["activity"], screen["description"] or "Unidentified"
                )

            experience_count = result.memory_summary.get("experience_count", 0)
            console.print(memory_table)
            console.print(
                f"[dim]Total learned experiences in brain:[/dim] [bold cyan]{experience_count}[/bold cyan]"
            )

        if not result.success:
            console.print(f"[bold red]Failure Reason:[/bold red] {result.completion_reason}")

        return 0 if result.success else 1

    async def explore(
        self,
        max_steps: int = 100,
        device_serial: Optional[str] = None,
        package_name: Optional[str] = None,
    ) -> int:
        """
        Run an exploration workflow with rich UI.
        """

        self.__setup_signals()

        scope_label = (
            f"[cyan]Package:[/cyan] {package_name}"
            if package_name
            else "[cyan]Package:[/cyan] auto-detect"
        )
        console.print(
            Panel.fit(
                f"[bold blue]Fathom Explorer[/bold blue]\n[cyan]Goal:[/cyan] Map application structure\n{scope_label}",
                border_style="blue",
            )
        )
        console.print("[dim]Pause/Resume: type 'p' + Enter or send SIGUSR1.[/dim]")

        pause_task: Optional[asyncio.Task[None]] = None
        try:
            pause_task = asyncio.create_task(self.__listen_for_pause())
            console.print("[bold green]Exploring...[/bold green]")
            result = await self.runner.run_exploration(
                max_steps=max_steps,
                device_serial=device_serial,
                package_name=package_name,
            )
        except (asyncio.CancelledError, KeyboardInterrupt):
            console.print("\n[bold red]Exploration cancelled by user.[/bold red]")
            return 1
        except FathomError as exception:
            logger.error(f"Exploration Error: {exception}")
            console.print(f"[bold red]Fathom Error:[/bold red] {exception}")
            return 1
        except Exception as exception:
            logger.exception("Unexpected error")
            console.print(f"[bold red]Unexpected Error:[/bold red] {exception}")
            return 1
        finally:
            if pause_task:
                pause_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await pause_task

        table = Table(title="Exploration Results", border_style="green")

        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")

        table.add_row("Unique Screens", str(result.unique_screens))
        table.add_row("Total Actions", str(result.total_actions))
        table.add_row("Total Transitions", str(result.total_transitions))
        table.add_row("Coverage", f"{result.coverage_percentage:.1f}%")

        console.print(table)

        if result.metrics:
            audit_table = Table(title="Timing Audit", border_style="blue")
            audit_table.add_column("Operation", style="cyan")
            audit_table.add_column("Total Time (s)", style="magenta", justify="right")
            audit_table.add_column("Avg/Step (s)", style="yellow", justify="right")

            token_metrics = result.metrics.get("Tokens")

            for operation, data in result.metrics.items():
                if operation == "Tokens":
                    continue

                total = data.get("total", 0.0)
                avg = data.get("avg", 0.0)
                audit_table.add_row(operation, f"{total:.2f}s", f"{avg:.2f}s")

            console.print(audit_table)

            if token_metrics:
                token_table = Table(title="Resource Usage (Tokens)", border_style="yellow")
                token_table.add_column("Metric", style="cyan")
                token_table.add_column("Value", style="magenta", justify="right")

                token_table.add_row("Prompt Tokens", f"{token_metrics.get('prompt', 0):,.0f}")
                token_table.add_row(
                    "Completion Tokens", f"{token_metrics.get('completion', 0):,.0f}"
                )
                token_table.add_row("Cached Tokens", f"{token_metrics.get('cached', 0):,.0f}")
                token_table.add_row("Total Tokens", f"{token_metrics.get('total', 0):,.0f}")

                console.print(token_table)

        # Display exploration insights report
        console.print("\n")
        self._display_exploration_insights(result)

        return 0

    def _display_exploration_insights(self, result: "ExplorationResult") -> None:
        """Display comprehensive exploration insights from knowledge graph analysis."""
        from fathom.services.exploration_report import ExplorationReportGenerator

        kg_json = result.knowledge_graph
        if not kg_json or not kg_json.get("nodes"):
            return

        # Rebuild knowledge graph from exported data (lightweight)
        from fathom.infrastructure.memory.knowledge_graph import KnowledgeGraph

        kg = KnowledgeGraph()

        # Manually populate nodes and edges from exported data
        for node_data in kg_json.get("nodes", []):
            from fathom.infrastructure.memory.knowledge_graph import GraphNode

            node = GraphNode(
                visual_hash=node_data["visual_hash"],
                activity=node_data["activity"],
                description=node_data.get("description"),
                first_seen=node_data.get("first_seen"),
                last_seen=node_data.get("last_seen"),
                visit_count=node_data.get("visit_count", 0),
            )
            kg._KnowledgeGraph__nodes[node.visual_hash] = node  # type: ignore[attr-defined]

        for edge_data in kg_json.get("edges", []):
            from fathom.infrastructure.memory.knowledge_graph import GraphEdge

            edge = GraphEdge(
                source_hash=edge_data["source_hash"],
                destination_hash=edge_data["destination_hash"],
                action_type=edge_data["action_type"],
                action_target=edge_data["action_target"],
                count=edge_data.get("count", 1),
            )
            kg._KnowledgeGraph__edges.setdefault(edge_data["source_hash"], []).append(edge)  # type: ignore[attr-defined]

        # Generate report generator
        report_gen = ExplorationReportGenerator(kg)

        # Detect cycles
        cycles = kg.detect_cycles()

        # Display insights
        insights_table = Table(title="Graph Analysis & Insights", border_style="magenta")
        insights_table.add_column("Metric", style="cyan")
        insights_table.add_column("Value", style="magenta")

        insights_table.add_row("Graph Diameter", str(kg.get_graph_diameter() or "N/A"))
        insights_table.add_row("Cycles Detected", str(len(cycles)))
        insights_table.add_row("Total Edges", str(kg.edge_count))

        console.print(insights_table)

        # Show critical screens
        critical = report_gen._identify_critical_screens()
        if critical:
            critical_table = Table(
                title="Critical Screens (Hubs & Bottlenecks)", border_style="yellow"
            )
            critical_table.add_column("Screen", style="cyan")
            critical_table.add_column("Type", style="yellow")
            critical_table.add_column("Connections", style="magenta", justify="right")

            for screen in critical[:5]:
                critical_table.add_row(
                    screen["name"][:40],
                    screen["type"],
                    str(screen["connectivity"]),
                )

            console.print(critical_table)

        # Show reachability analysis
        reachability = report_gen._analyze_reachability()
        if reachability:
            reach_table = Table(
                title="Reachability Analysis from Major Screens", border_style="cyan"
            )
            reach_table.add_column("Screen", style="cyan")
            reach_table.add_column("Forward", style="magenta", justify="right")
            reach_table.add_column("Backward", style="yellow", justify="right")

            for screen_name, reach_data in reachability.items():
                reach_table.add_row(
                    screen_name[:40],
                    reach_data["forward_coverage"],
                    str(reach_data["backward_reach"]),
                )

            console.print(reach_table)

        # Show recommendations
        recommendations = report_gen._generate_recommendations(
            result.knowledge_graph["stats"], cycles
        )
        if recommendations:
            rec_panel = Panel(
                "\n".join(recommendations),
                title="[bold]Recommendations[/bold]",
                border_style="green",
            )
            console.print(rec_panel)


def main() -> int:
    """
    CLI entry point.
    """

    parser = argparse.ArgumentParser(description="Fathom: AI-powered mobile automation agent")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    run_parser = subparsers.add_parser("run", help="Run an intent-based automation workflow")
    run_parser.add_argument("intent", type=str, help="The goal description")
    run_parser.add_argument("--serial", "-s", type=str, help="Device serial number")
    run_parser.add_argument("--api-key", "-k", type=str, help="Gemini API Key")
    run_parser.add_argument(
        "--max-steps",
        "-ms",
        type=int,
        default=None,
        help="Maximum steps (default: MAX_STEPS env or 100)",
    )
    run_parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    run_parser.add_argument("--use-xml", "-x", action="store_true", help="Use XML bounding boxes")
    run_parser.add_argument(
        "--prompt-version",
        type=str,
        default=None,
        help="Version of prompt/toolset to use",
    )
    run_parser.add_argument(
        "--hitl",
        action="store_true",
        default=False,
        help="Require human approval before executing device actions",
    )
    explore_parser = subparsers.add_parser("explore", help="Run app exploration")
    explore_parser.add_argument(
        "--package", "-p", type=str, help="Target package name to explore (e.g. com.example.app)"
    )
    explore_parser.add_argument(
        "--max-steps",
        "-ms",
        type=int,
        default=None,
        help="Maximum exploration steps (default: EXPLORE_MAX_STEPS env or 50)",
    )
    explore_parser.add_argument("--serial", "-s", type=str, help="Device serial number")
    explore_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )

    args = parser.parse_args()
    settings = FathomSettings()

    if hasattr(args, "api_key") and args.api_key:
        settings.gemini_api_key = args.api_key

    if hasattr(args, "verbose") and args.verbose:
        settings.log_level = "DEBUG"
    else:
        settings.log_level = "WARNING"

    BaseLogger.configure(settings)

    getLogger("PIL").setLevel("ERROR")
    getLogger("urllib3").setLevel("ERROR")
    getLogger("httpx").setLevel("WARNING")
    getLogger("google").setLevel("WARNING")
    getLogger("google.auth").setLevel("WARNING")

    try:
        cli = FathomCLI(settings)
    except FathomError as exception:
        console.print(f"[bold red]Configuration Error:[/bold red] {exception}")
        return 1

    try:
        if args.command == "run":
            run_steps = args.max_steps if args.max_steps is not None else settings.max_steps
            result = asyncio.run(
                cli.run(
                    intent=args.intent,
                    use_xml=args.use_xml,
                    max_steps=run_steps,
                    device_serial=args.serial,
                    prompt_version=args.prompt_version,
                    human_in_loop=args.hitl,
                )
            )
            return result
        elif args.command == "explore":
            explore_steps = (
                args.max_steps if args.max_steps is not None else settings.explore_max_steps
            )
            return asyncio.run(
                cli.explore(
                    max_steps=explore_steps,
                    device_serial=args.serial,
                    package_name=args.package,
                )
            )
        else:
            parser.print_help()
            return 0
    except KeyboardInterrupt:
        # Final safety catch for KeyboardInterrupt at the top level
        return 1


if __name__ == "__main__":
    sys.exit(main())
