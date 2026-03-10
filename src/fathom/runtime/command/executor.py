from __future__ import annotations

import asyncio
import signal
import time
from logging import getLogger
from typing import Optional

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from fathom.adapters.telemetry.console import ConsoleTelemetryAdapter
from fathom.adapters.telemetry.structlog import StructlogAdapter
from fathom.base.paths import SharedPathManager
from fathom.core.exceptions import FathomError
from fathom.interfaces.factory import (
    DeviceFactoryPort,
    LLMFactoryPort,
    PerceptionFactoryPort,
    SignalFactoryPort,
)
from fathom.interfaces.signal import SignalPort
from fathom.runtime.builder import Fathom
from fathom.runtime.command.resolver import (
    RuntimeDeviceDefaultsResolver,
    RuntimeDeviceDefaultsResolverPort,
)
from fathom.runtime.factories import DeviceFactory, LLMFactory, PerceptionFactory, SignalFactory
from fathom.runtime.runner import FathomRunner
from fathom.schemas.configuration import DeviceConfiguration, LLMConfiguration
from fathom.schemas.orchestration import WorkflowRequest
from fathom.schemas.results import IntentResult
from fathom.settings.env import FathomSettings

console = Console()
logger = getLogger(__name__)


class CommandExecutor:
    """
    Runtime executor for CLI command workflows.
    """

    def __init__(
        self,
        *,
        settings: FathomSettings,
        llm_factory: LLMFactoryPort | None = None,
        device_factory: DeviceFactoryPort | None = None,
        perception_factory: PerceptionFactoryPort | None = None,
        signal_factory: SignalFactoryPort | None = None,
        device_defaults_resolver: RuntimeDeviceDefaultsResolverPort | None = None,
    ) -> None:
        """
        Initialize executor with settings.
        """

        self.__cancelled = False
        self.__settings = settings
        self.__runner: Optional[FathomRunner] = None
        self.__device_factory = device_factory or DeviceFactory()
        self.__llm_factory = llm_factory or LLMFactory()
        self.__perception_factory = perception_factory or PerceptionFactory()
        self.__signal_factory = signal_factory or SignalFactory()
        self.__device_defaults_resolver = device_defaults_resolver or RuntimeDeviceDefaultsResolver(
            settings=settings
        )

    def __setup_signals(self) -> None:
        """
        Configure signal handlers for graceful shutdown.
        """

        for os_signal in (signal.SIGINT, signal.SIGTERM):
            try:
                loop = asyncio.get_running_loop()
                loop.add_signal_handler(os_signal, self.__handle_interrupt)
            except (NotImplementedError, ValueError, RuntimeError) as exception:
                logger.debug("Signal handler registration skipped for %s: %s", os_signal, exception)

    def __handle_interrupt(self) -> None:
        """
        Handle process interrupt and cancel current runner.
        """

        console.print("\n[bold yellow]Stopping gracefully... Please wait.[/bold yellow]")
        self.__cancelled = True

        if self.__runner:
            self.__runner.cancel()

    def __build_llm_configuration(self) -> LLMConfiguration:
        """
        Build LLM configuration from runtime settings.
        """

        return LLMConfiguration(
            model=self.__settings.gemini_model,
            api_key=self.__settings.gemini_api_key,
            location=self.__settings.vertex_location,
            project_id=self.__settings.vertex_project_id,
            use_cache=getattr(self.__settings, "use_cache", True),
            credentials=self.__settings.google_application_credentials,
        )

    def __resolve_device_configuration(self, *, request: WorkflowRequest) -> DeviceConfiguration:
        """
        Resolve request device configuration with safe defaults.
        """
        return self.__device_defaults_resolver.resolve(configuration=request.device)

    def __create_signal_adapter(self, *, request: WorkflowRequest) -> SignalPort:
        """
        Create signal adapter from request interaction mode.
        """

        signal_adapter = self.__signal_factory.create(
            interactive=request.interactive,
            signal_type=request.signal_type,
        )

        if request.interactive:
            console.print("[bold cyan]Interactive mode enabled[/bold cyan]")
            console.print("[dim]Agent will ask questions when uncertain[/dim]")

        return signal_adapter

    def __create_runner(
        self, *, request: WorkflowRequest, signal_adapter: SignalPort
    ) -> FathomRunner:
        """
        Build configured runtime runner for command execution.
        """

        path_manager = SharedPathManager(settings=self.__settings)
        llm_configuration = self.__build_llm_configuration()
        device_configuration = self.__resolve_device_configuration(request=request)
        device_adapter = self.__device_factory.create(configuration=device_configuration)
        perception_adapter = self.__perception_factory.create(
            configuration=device_configuration,
            device=device_adapter,
        )
        llm_adapter = self.__llm_factory.create(configuration=llm_configuration)
        telemetry_adapter = ConsoleTelemetryAdapter(
            inner=StructlogAdapter(),
            console=console,
        )

        return (
            Fathom.builder(path_manager=path_manager)
            .with_device(port=device_adapter)
            .with_perception(port=perception_adapter)
            .with_llm(port=llm_adapter)
            .with_signal(port=signal_adapter)
            .with_telemetry(port=telemetry_adapter)
            .build()
        )

    async def __run_intent_workflow(self, *, request: WorkflowRequest) -> IntentResult:
        """
        Execute intent workflow with spinner in non-interactive mode.
        """

        if self.__runner is None:
            raise FathomError("Runner is not initialized")

        if request.interactive:
            return await self.__runner.run_intent(
                intent=request.intent,
                use_xml=request.use_xml,
                max_steps=request.max_steps,
                request_id=request.session_id,
                realignment=request.realignment,
            )

        with console.status("[bold green]Agent working...[/bold green]\n", spinner="dots"):
            return await self.__runner.run_intent(
                intent=request.intent,
                use_xml=request.use_xml,
                max_steps=request.max_steps,
                request_id=request.session_id,
                realignment=request.realignment,
            )

    def __print_execution_summary(self, *, result: IntentResult) -> None:
        """
        Render execution summary and auxiliary tables.
        """

        table = Table(title="Execution Summary", border_style="green" if result.success else "red")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        table.add_row("Status", "[green]Success[/green]" if result.success else "[red]Failed[/red]")
        table.add_row("Reason", escape(result.completion_reason or "N/A"))
        table.add_row("Steps Taken", str(result.steps_taken))
        console.print(table)

        self.__print_timing_audit(result=result)
        self.__print_memory_summary(result=result)

        if not result.success:
            console.print(
                f"[bold red]Failure Reason:[/bold red] {escape(result.completion_reason)}"
            )

    def __print_timing_audit(self, *, result: IntentResult) -> None:
        """
        Render timing table and token usage.
        """

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

            if isinstance(data, dict):
                total = data.get("total", 0.0)
                average = data.get("avg", 0.0)
                audit_table.add_row(operation, f"{total:.2f}s", f"{average:.2f}s")

        console.print(audit_table)
        self.__print_token_metrics(token_metrics=token_metrics)

    def __print_token_metrics(self, *, token_metrics: object) -> None:
        """
        Render token usage table when token metrics exist.
        """

        if not isinstance(token_metrics, dict):
            return

        token_table = Table(title="Resource Usage (Tokens)", border_style="yellow")
        token_table.add_column("Metric", style="cyan")
        token_table.add_column("Value", style="magenta", justify="right")
        token_table.add_row("Prompt Tokens", f"{token_metrics.get('prompt', 0):,.0f}")
        token_table.add_row("Completion Tokens", f"{token_metrics.get('completion', 0):,.0f}")
        token_table.add_row("Cached Tokens", f"{token_metrics.get('cached', 0):,.0f}")
        token_table.add_row("Total Tokens", f"{token_metrics.get('total', 0):,.0f}")
        console.print(token_table)

    def __print_memory_summary(self, *, result: IntentResult) -> None:
        """
        Render memory summary table for learned screens.
        """

        if not result.memory_summary:
            return

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

    async def __cleanup_runner(self) -> None:
        """
        Cleanup current runner if initialized.
        """

        if self.__runner:
            await self.__runner.cleanup()

    async def run(self, *, request: WorkflowRequest) -> int:
        """
        Execute intent command flow.
        """

        self.__setup_signals()

        console.print(
            Panel.fit(
                f"[bold blue]Fathom Agent[/bold blue]\n[cyan]Intent:[/cyan] {request.intent}",
                border_style="blue",
            )
        )

        try:
            signal_adapter = self.__create_signal_adapter(request=request)
            self.__runner = self.__create_runner(request=request, signal_adapter=signal_adapter)
            result = await self.__run_intent_workflow(request=request)
            self.__print_execution_summary(result=result)
            return 0 if result.success else 1

        except (asyncio.CancelledError, KeyboardInterrupt):
            console.print("\n[bold red]Execution cancelled by user.[/bold red]")
            return 1
        except FathomError as exception:
            logger.error("CLI error: %s", exception)
            console.print(f"[bold red]Fathom Error:[/bold red] {escape(str(exception))}")
            return 1
        except Exception as exception:
            logger.exception("Unexpected error")
            console.print(f"[bold red]Unexpected Error:[/bold red] {escape(str(exception))}")
            return 1
        finally:
            await self.__cleanup_runner()

    async def explore(self, *, request: WorkflowRequest) -> int:
        """
        Execute exploration command flow.
        """

        self.__setup_signals()

        console.print(
            Panel.fit(
                "[bold blue]Fathom Explorer[/bold blue]\n[cyan]Goal:[/cyan] Map application structure",
                border_style="blue",
            )
        )

        start_time = time.time()

        try:
            signal_adapter = self.__signal_factory.create(
                interactive=False,
                signal_type=request.signal_type,
            )
            self.__runner = self.__create_runner(request=request, signal_adapter=signal_adapter)

            with console.status("[bold green]Exploring...[/bold green]", spinner="earth"):
                result = await self.__runner.run_exploration(
                    max_steps=request.max_steps,
                    request_id=request.session_id,
                )

            table = Table(title="Exploration Results", border_style="green")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="magenta")
            table.add_row("Unique Screens", str(result.unique_screens))
            table.add_row("Total Actions", str(result.total_actions))
            table.add_row("Total Transitions", str(result.total_transitions))
            table.add_row("Coverage", f"{result.coverage_percentage:.1f}%")
            table.add_row("Duration", f"{(time.time() - start_time):.2f}s")
            console.print(table)

            return 0

        except (asyncio.CancelledError, KeyboardInterrupt):
            console.print("\n[bold red]Exploration cancelled by user.[/bold red]")
            return 1
        except Exception as exception:
            logger.exception("Unexpected exploration error")
            console.print(f"[bold red]Unexpected Error:[/bold red] {escape(str(exception))}")
            return 1
        finally:
            await self.__cleanup_runner()
