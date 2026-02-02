import argparse
import asyncio
import sys
from logging import getLogger
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from fathom.base.logger import BaseLogger
from fathom.exceptions import FathomError
from fathom.services.runner import FathomRunner
from fathom.settings.env import FathomSettings

logger = getLogger(__name__)
console = Console()


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

    async def run(
        self,
        intent: str,
        max_steps: int = 20,
        device_serial: Optional[str] = None,
    ) -> int:
        """
        Run an intent workflow with rich UI.
        """

        # Header
        console.print(
            Panel.fit(
                f"[bold blue]Fathom Agent[/bold blue]\n[cyan]Intent:[/cyan] {intent}",
                border_style="blue",
            )
        )

        try:
            # Execution with Spinner
            with console.status("[bold green]Agent working...[/bold green]", spinner="dots"):
                result = await self.runner.run_intent(
                    intent=intent, device_serial=device_serial, max_steps=max_steps
                )

            # Results Table
            table = Table(
                title="Execution Summary", border_style="green" if result.success else "red"
            )
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="magenta")

            table.add_row(
                "Status", "[green]Success[/green]" if result.success else "[red]Failed[/red]"
            )
            table.add_row("Reason", result.completion_reason)
            table.add_row("Steps Taken", str(result.steps_taken))

            console.print(table)

            if not result.success:
                console.print(f"[bold red]Failure Reason:[/bold red] {result.completion_reason}")

            return 0 if result.success else 1

        except FathomError as exception:
            logger.error(f"CLI Error: {exception}")
            console.print(f"[bold red]Fathom Error:[/bold red] {exception}")
            return 1
        except Exception as exception:
            logger.exception("Unexpected error")
            console.print(f"[bold red]Unexpected Error:[/bold red] {exception}")
            return 1


def main() -> int:
    """
    CLI entry point.
    """

    parser = argparse.ArgumentParser(description="Fathom: AI-powered mobile automation agent")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # 'run' command
    run_parser = subparsers.add_parser("run", help="Run an intent-based automation workflow")
    run_parser.add_argument("intent", type=str, help="The goal description (e.g., 'Open settings')")
    run_parser.add_argument("--serial", "-s", type=str, help="Device serial number (overrides env)")
    run_parser.add_argument("--api-key", "-k", type=str, help="Gemini API Key (overrides env)")
    run_parser.add_argument("--max-steps", type=int, default=20, help="Maximum steps allowed")
    run_parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    # Load settings
    settings = FathomSettings()

    # CLI overrides
    if hasattr(args, "api_key") and args.api_key:
        settings.gemini_api_key = args.api_key

    if hasattr(args, "verbose") and args.verbose:
        settings.log_level = "DEBUG"

    BaseLogger.configure(settings)

    try:
        cli = FathomCLI(settings)
    except FathomError as exception:
        console.print(f"[bold red]Configuration Error:[/bold red] {exception}")
        return 1

    if args.command == "run":
        return asyncio.run(
            cli.run(
                intent=args.intent,
                max_steps=args.max_steps,
                device_serial=args.serial,
            )
        )
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
