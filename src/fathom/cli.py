import argparse
import asyncio
import signal
import sys
from logging import getLogger
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from fathom.base.logger import BaseLogger
from fathom.exceptions import FathomError
from fathom.orchestration.runner import FathomRunner
from fathom.runtime.command.explorer_tui import ExplorerApp
from fathom.runtime.command.wizard import InteractiveWizard, wizard_argv
from fathom.schemas.results import ExplorationResult
from fathom.settings.env import FathomSettings

console = Console()
logger = getLogger(__name__)


class FathomCLI:
    """
    Fathom CLI — exploration-only entry point.
    """

    def __init__(self, settings: FathomSettings) -> None:
        self.settings = settings
        self.runner = FathomRunner(settings)

    def __setup_signals(self) -> None:
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop = asyncio.get_running_loop()
                loop.add_signal_handler(sig, self.__handle_interrupt)
            except (NotImplementedError, ValueError, RuntimeError):
                pass

    def __handle_interrupt(self) -> None:
        console.print("\n[bold yellow]Stopping gracefully... Please wait.[/bold yellow]")
        self.runner.cancel()

    async def explore(
        self,
        max_steps: int = 50,
        device_serial: Optional[str] = None,
        package_name: Optional[str] = None,
        focus: Optional[str] = None,
        tui: bool = False,
    ) -> int:
        """
        Run an exploration workflow.

        ``tui=True`` renders inside a full-screen Textual app with a live
        scrollable event feed. ``tui=False`` keeps the legacy spinner +
        results-table flow.
        """

        if tui:
            return await self.__explore_tui(
                max_steps=max_steps,
                device_serial=device_serial,
                package_name=package_name,
                focus=focus,
            )

        self.__setup_signals()

        package_label = package_name or "auto-detect"
        focus_line = f"\n[cyan]Focus:[/cyan] {focus}" if focus and focus.strip() else ""
        console.print(
            Panel.fit(
                f"[bold blue]Fathom Explorer[/bold blue]\n"
                f"[cyan]Package:[/cyan] {package_label}\n"
                f"[cyan]Max steps:[/cyan] {max_steps}"
                f"{focus_line}",
                border_style="blue",
            )
        )

        try:
            with console.status("[bold green]Exploring...[/bold green]", spinner="earth"):
                result = await self.runner.run_exploration(
                    max_steps=max_steps,
                    device_serial=device_serial,
                    package_name=package_name,
                    focus=focus,
                )

            self.__print_results(result=result)
            return 0 if result.success else 1

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

    async def __explore_tui(
        self,
        *,
        max_steps: int,
        device_serial: Optional[str],
        package_name: Optional[str],
        focus: Optional[str],
    ) -> int:
        """
        Run exploration inside the live Textual TUI.

        The agent runs on a worker thread inside ``ExplorerApp``; events
        emitted by the graph are routed back to the TUI via
        ``app.record_event``. After the app exits we fall back to the
        regular console for the final results / timing tables so the
        operator gets the same end-of-run summary as the legacy path.
        """

        result_holder: dict[str, Optional[ExplorationResult]] = {"result": None}
        error_holder: dict[str, Optional[BaseException]] = {"error": None}

        async def workflow() -> bool:
            try:
                result = await self.runner.run_exploration(
                    max_steps=max_steps,
                    device_serial=device_serial,
                    package_name=package_name,
                    focus=focus,
                    event_bus=app.record_event,
                )
            except BaseException as exc:
                error_holder["error"] = exc
                raise
            result_holder["result"] = result
            return bool(result.success)

        app = ExplorerApp(
            package=package_name or "auto-detect",
            max_steps=max_steps,
            workflow=workflow,
            on_cancel=self.runner.cancel,
        )

        try:
            await app.run_async()
        except (asyncio.CancelledError, KeyboardInterrupt):
            self.runner.cancel()
            return 1
        except Exception as exception:  # noqa: BLE001
            logger.exception("Explorer TUI crashed")
            console.print(f"[bold red]TUI Error:[/bold red] {exception}")
            self.runner.cancel()
            return 1

        # Belt-and-braces: whatever the exit path out of the TUI, make
        # sure the agent's cancel flag is set so an in-flight LLM call
        # doesn't linger past the app.
        self.runner.cancel()

        # Give the cancelled workflow a brief window to finalize (write
        # the exploration report, emit the terminal event). The worker
        # thread is a daemon so it can't hang the process.
        deadline = asyncio.get_event_loop().time() + 5.0
        while result_holder["result"] is None and error_holder["error"] is None:
            if asyncio.get_event_loop().time() >= deadline:
                break
            await asyncio.sleep(0.1)

        result = result_holder["result"]
        if result is not None:
            self.__print_results(result=result)
            return 0 if result.success else 1

        error = error_holder["error"]
        if error is not None:
            console.print(f"[bold red]Exploration Error:[/bold red] {error}")
        return app.exit_code

    def __print_results(self, *, result: ExplorationResult) -> None:
        """Render the post-run results / timing / token tables to the console."""

        table = Table(
            title="Exploration Results",
            border_style="green" if result.success else "red",
        )
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")

        table.add_row(
            "Status",
            "[green]Success[/green]" if result.success else "[red]Failed[/red]",
        )
        table.add_row("Completion", result.completion_reason)
        table.add_row("Unique Screens", str(result.unique_screens))
        table.add_row("Total Actions", str(result.total_actions))
        table.add_row("Total Transitions", str(result.total_transitions))
        table.add_row("Coverage", f"{result.coverage_percentage:.1f}%")

        if result.discovered_activities:
            table.add_row("Activities", ", ".join(result.discovered_activities[:10]))

        console.print(table)

        if not result.metrics:
            return

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
            token_table.add_row("Completion Tokens", f"{token_metrics.get('completion', 0):,.0f}")
            token_table.add_row("Cached Tokens", f"{token_metrics.get('cached', 0):,.0f}")
            token_table.add_row("Total Tokens", f"{token_metrics.get('total', 0):,.0f}")

            console.print(token_table)


def main() -> int:
    """
    CLI entry point.
    """

    parser = argparse.ArgumentParser(description="Fathom: AI-powered mobile app explorer")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    explore_parser = subparsers.add_parser("explore", help="Run app exploration")
    explore_parser.add_argument(
        "--package", "-p", type=str, default=None, help="Target app package name"
    )
    explore_parser.add_argument("--max-steps", type=int, default=50, help="Maximum steps allowed")
    explore_parser.add_argument("--serial", "-s", type=str, default=None, help="Device serial")
    explore_parser.add_argument(
        "--focus",
        "-f",
        type=str,
        default=None,
        help=(
            "Focus exploration on a specific section or flow (e.g. "
            "'the checkout flow', 'profile settings'). Omit for full-breadth mapping."
        ),
    )
    explore_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )
    explore_parser.add_argument(
        "--tui",
        action="store_true",
        help="Run inside a live Textual TUI (scrollable event feed)",
    )

    args = parser.parse_args()

    if not args.command:
        try:
            wizard_result = InteractiveWizard(console=console).run()
        except KeyboardInterrupt:
            return 1
        if wizard_result is None:
            console.print("[yellow]Aborted.[/yellow]")
            return 0
        args = parser.parse_args(wizard_argv(wizard_result))

    settings = FathomSettings()

    if hasattr(args, "verbose") and args.verbose:
        settings.log_level = "DEBUG"

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
        if args.command == "explore":
            return asyncio.run(
                cli.explore(
                    max_steps=args.max_steps,
                    device_serial=args.serial,
                    package_name=args.package,
                    focus=args.focus,
                    tui=getattr(args, "tui", False),
                )
            )
        else:
            parser.print_help()
            return 0
    except KeyboardInterrupt:
        return 1


if __name__ == "__main__":
    sys.exit(main())
