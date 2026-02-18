from __future__ import annotations

import argparse
import asyncio
import signal
import sys
from logging import getLogger
from typing import Optional

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from fathom.adapters.llm.gemini import GeminiLLM
from fathom.adapters.signal.interactive import InteractiveSignal
from fathom.adapters.signal.noop import NoopSignal
from fathom.adapters.signal.socket import SocketSignal
from fathom.base.logger import BaseLogger
from fathom.base.paths import SharedPathManager
from fathom.core.exceptions import FathomError
from fathom.interfaces.signal import SignalPort
from fathom.runtime.builder import Fathom
from fathom.runtime.factories import DeviceFactory
from fathom.runtime.runner import FathomRunner
from fathom.schemas.configuration import DeviceConfiguration, LLMConfiguration
from fathom.schemas.orchestration import RealignmentPolicy, WorkflowRequest
from fathom.settings.env import FathomSettings

console = Console()
logger = getLogger(__name__)


class FathomCLI:
    """
    Fathom CLI application using hexagonal architecture.
    """

    def __init__(self, settings: FathomSettings) -> None:
        """
        Initialize CLI with settings.
        """

        self.__cancelled = False
        self.__settings = settings
        self.__runner: Optional[FathomRunner] = None

    def __setup_signals(self) -> None:
        """
        Configure signal handlers for graceful shutdown.
        """

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop = asyncio.get_running_loop()
                loop.add_signal_handler(sig, self.__handle_interrupt)
            except (NotImplementedError, ValueError, RuntimeError):
                pass

    def __handle_interrupt(self) -> None:
        """
        Handler called when a signal is received.
        """

        console.print("\n[bold yellow]Stopping gracefully... Please wait.[/bold yellow]")
        self.__cancelled = True

        if self.__runner:
            self.__runner.cancel()

    async def run(self, request: WorkflowRequest) -> int:
        """
        Run an intent workflow with rich UI.
        """

        self.__setup_signals()

        console.print(
            Panel.fit(
                f"[bold blue]Fathom Agent[/bold blue]\n[cyan]Intent:[/cyan] {request.intent}",
                border_style="blue",
            )
        )

        signal_adapter: SignalPort
        interactive_mode = request.interactive

        try:
            # Initialize shared paths
            path_manager = SharedPathManager(settings=self.__settings)

            # Build configurations
            serial = request.device_serial or self.__settings.android_serial
            device_configuration = DeviceConfiguration(type="LOCAL", serial_number=serial)

            llm_configuration = LLMConfiguration(
                model=self.__settings.gemini_model,
                api_key=self.__settings.gemini_api_key,
                location=self.__settings.vertex_location,
                project_id=self.__settings.vertex_project_id,
                credentials=self.__settings.google_application_credentials,
            )

            if interactive_mode:
                signal_type = getattr(request, "signal_type", "interactive")

                if signal_type == "socket":
                    signal_adapter = SocketSignal()
                    console.print("[bold cyan]🔌 Socket-based control enabled[/bold cyan]")
                else:
                    signal_adapter = InteractiveSignal()
                    console.print("[bold cyan]🤝 Interactive mode enabled[/bold cyan]")

                console.print("[dim]Agent will ask questions when uncertain[/dim]")
            else:
                signal_adapter = NoopSignal()

            # Create device adapter via factory
            device_adapter = DeviceFactory.create(configuration=device_configuration)

            # Use builder defaults for standard infrastructure (Memory, Storage, Telemetry)
            self.__runner = (
                Fathom.builder(path_manager=path_manager)
                .with_device(port=device_adapter)
                .with_llm(port=GeminiLLM(configuration=llm_configuration))
                .with_signal(port=signal_adapter)
                .build()
            )

            # Don't use spinner in interactive mode - it blocks output
            if interactive_mode:
                result = await self.__runner.run_intent(
                    intent=request.intent,
                    use_xml=request.use_xml,
                    max_steps=request.max_steps,
                    request_id=request.session_id,
                    realignment=request.realignment,
                )
            else:
                with console.status("[bold green]Agent working...[/bold green]\n", spinner="dots"):
                    result = await self.__runner.run_intent(
                        intent=request.intent,
                        use_xml=request.use_xml,
                        max_steps=request.max_steps,
                        request_id=request.session_id,
                        realignment=request.realignment,
                    )

            # Execution Summary
            table = Table(
                title="Execution Summary", border_style="green" if result.success else "red"
            )
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="magenta")

            table.add_row(
                "Status", "[green]Success[/green]" if result.success else "[red]Failed[/red]"
            )
            table.add_row("Reason", escape(result.completion_reason or "N/A"))
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

                    if isinstance(data, dict):
                        total = data.get("total", 0.0)
                        avg = data.get("avg", 0.0)
                        audit_table.add_row(operation, f"{total:.2f}s", f"{avg:.2f}s")

                console.print(audit_table)

                if token_metrics and isinstance(token_metrics, dict):
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

            # Memory Graph Summary
            if result.memory_summary:
                memory_table = Table(title="Agent Knowledge Graph (Brain)", border_style="cyan")
                memory_table.add_column("Screen Hash", style="dim")
                memory_table.add_column("Activity")
                memory_table.add_column("Agent Description", style="italic")

                screens = result.memory_summary.get("screens", [])
                if isinstance(screens, list):
                    for screen in screens[:10]:
                        memory_table.add_row(
                            screen.get("hash", ""),
                            screen.get("activity", ""),
                            screen.get("description") or "Unidentified",
                        )

                experience_count = result.memory_summary.get("experience_count", 0)
                console.print(memory_table)
                console.print(
                    f"[dim]Total learned experiences in brain:[/dim] [bold cyan]{experience_count}[/bold cyan]"
                )

            if not result.success:
                console.print(
                    f"[bold red]Failure Reason:[/bold red] {escape(result.completion_reason)}"
                )

            if self.__runner:
                await self.__runner.cleanup()
            return 0 if result.success else 1

        except (asyncio.CancelledError, KeyboardInterrupt):
            console.print("\n[bold red]Execution cancelled by user.[/bold red]")

            if self.__runner:
                await self.__runner.cleanup()

            return 1
        except FathomError as exception:
            logger.error(f"CLI Error: {exception}")
            console.print(f"[bold red]Fathom Error:[/bold red] {escape(str(exception))}")

            if self.__runner:
                await self.__runner.cleanup()

            return 1
        except Exception as exception:
            logger.exception("Unexpected error")
            console.print(f"[bold red]Unexpected Error:[/bold red] {escape(str(exception))}")

            if self.__runner:
                await self.__runner.cleanup()

            return 1

    async def explore(self, request: WorkflowRequest) -> int:
        """
        Run an exploration workflow with rich UI.
        """

        self.__setup_signals()

        console.print(
            Panel.fit(
                "[bold blue]Fathom Explorer[/bold blue]\n[cyan]Goal:[/cyan] Map application structure",
                border_style="blue",
            )
        )

        try:
            serial = request.device_serial or self.__settings.android_serial

            # Initialize shared paths
            path_manager = SharedPathManager(settings=self.__settings)

            llm_configuration = LLMConfiguration(
                model=self.__settings.gemini_model,
                api_key=self.__settings.gemini_api_key,
                location=self.__settings.vertex_location,
                project_id=self.__settings.vertex_project_id,
                credentials=self.__settings.google_application_credentials,
            )

            device_configuration = DeviceConfiguration(type="LOCAL", serial_number=serial)

            # Create device adapter via factory
            device_adapter = DeviceFactory.create(configuration=device_configuration)

            self.__runner = (
                Fathom.builder(path_manager=path_manager)
                .with_device(port=device_adapter)
                .with_llm(port=GeminiLLM(configuration=llm_configuration))
                .with_signal(port=NoopSignal())
                .build()
            )

            with console.status("[bold green]Exploring...[/bold green]", spinner="earth"):
                result = await self.__runner.run_exploration(
                    max_steps=request.max_steps, request_id=request.session_id
                )

            table = Table(title="Exploration Results", border_style="green")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="magenta")

            table.add_row("Unique Screens", str(result.unique_screens))
            table.add_row("Total Actions", str(result.total_actions))
            table.add_row("Total Transitions", str(result.total_transitions))
            table.add_row("Coverage", f"{result.coverage_percentage:.1f}%")

            console.print(table)

            if self.__runner:
                await self.__runner.cleanup()

            return 0

        except (asyncio.CancelledError, KeyboardInterrupt):
            console.print("\n[bold red]Exploration cancelled by user.[/bold red]")

            if self.__runner:
                await self.__runner.cleanup()

            return 1
        except Exception as exception:
            logger.exception("Unexpected error")
            console.print(f"[bold red]Unexpected Error:[/bold red] {escape(str(exception))}")

            if self.__runner:
                await self.__runner.cleanup()

            return 1


def main() -> int:
    """
    CLI entry point.
    """

    parser = argparse.ArgumentParser(
        description="Fathom: AI-powered mobile automation agent", allow_abbrev=False
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    run_parser = subparsers.add_parser("run", help="Run an intent-based automation workflow")
    run_parser.add_argument("intent", type=str, help="The goal description")
    run_parser.add_argument("--serial", "-s", type=str, help="Device serial number")
    run_parser.add_argument("--api-key", "-k", type=str, help="Gemini API Key")
    run_parser.add_argument("--max-steps", type=int, default=20, help="Maximum steps allowed")
    run_parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    run_parser.add_argument("--use-xml", "-x", action="store_true", help="Use XML bounding boxes")
    run_parser.add_argument(
        "--interactive", "-i", action="store_true", help="Enable interactive HITL mode"
    )
    run_parser.add_argument(
        "--signal",
        type=str,
        default="interactive",
        choices=["interactive", "socket"],
        help="Type of signal adapter to use in interactive mode",
    )
    run_parser.add_argument(
        "--realignment-budget",
        type=int,
        default=3,
        help="Maximum allowed consecutive re-plans",
    )
    run_parser.add_argument(
        "--no-realignment",
        action="store_false",
        dest="immediate_realignment",
        help="Disable immediate re-planning on context injection",
    )
    run_parser.set_defaults(immediate_realignment=True)

    explore_parser = subparsers.add_parser("explore", help="Run app exploration")
    explore_parser.add_argument("--max-steps", type=int, default=50, help="Maximum steps allowed")
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

    BaseLogger.configure(settings)

    getLogger("PIL").setLevel("ERROR")
    getLogger("urllib3").setLevel("ERROR")
    getLogger("httpx").setLevel("WARNING")
    getLogger("google").setLevel("WARNING")
    getLogger("google.auth").setLevel("WARNING")

    try:
        cli = FathomCLI(settings)
    except FathomError as exception:
        console.print(f"[bold red]Configuration Error:[/bold red] {escape(str(exception))}")
        return 1

    try:
        if args.command == "run":
            realignment = RealignmentPolicy(
                immediate=args.immediate_realignment, budget=args.realignment_budget
            )

            request = WorkflowRequest(
                intent=args.intent,
                use_xml=args.use_xml,
                signal_type=args.signal,
                realignment=realignment,
                max_steps=args.max_steps,
                device_serial=args.serial,
                interactive=args.interactive,
            )
            result = asyncio.run(cli.run(request=request))
            return result

        elif args.command == "explore":
            request = WorkflowRequest(
                max_steps=args.max_steps,
                device_serial=args.serial,
                intent="Explore application structure",
            )
            return asyncio.run(cli.explore(request=request))
        else:
            parser.print_help()
            return 0
    except KeyboardInterrupt:
        return 1


if __name__ == "__main__":
    sys.exit(main())
