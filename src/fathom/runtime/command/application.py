from __future__ import annotations

import argparse
import asyncio
import datetime
from logging import getLogger
from pathlib import Path
from typing import Dict, Optional, Tuple

from pydantic import ValidationError
from rich.console import Console
from rich.markup import escape

from fathom.base.logger import BaseLogger
from fathom.constants.exploration import DEFAULT_EXPLORATION_INTENT
from fathom.constants.run import SignalAdapterType, TargetKind
from fathom.core.exceptions import FathomError
from fathom.runtime.command.executor import CommandExecutor
from fathom.runtime.command.resolver import (
    LocalDeviceConfigurationResolver,
    LocalDeviceConfigurationResolverPort,
)
from fathom.schemas.cli import ExploreCommandInput, LocalCommandInput, RunCommandInput
from fathom.schemas.configuration import DeviceConfiguration
from fathom.schemas.run import (
    ExplorationObjectiveConfiguration,
    ExplorationRunRequest,
    IntentObjectiveConfiguration,
    IntentRunRequest,
    InteractionConfiguration,
    MemoryConfiguration,
    ModelSelectionConfiguration,
    RealignmentPolicy,
    ResourceConfiguration,
    RunMetadata,
    RuntimeConfiguration,
    TargetConfiguration,
)
from fathom.settings.env import FathomSettings

console = Console()
logger = getLogger(__name__)


class CommandApplication:
    """
    CLI command application for request parsing and dispatch.
    """

    def __init__(
        self,
        *,
        local_device_resolver: LocalDeviceConfigurationResolverPort | None = None,
    ) -> None:
        """
        Initialize command application parser.
        """

        self.__local_device_resolver = local_device_resolver or LocalDeviceConfigurationResolver()
        self.__parser = self.__build_parser()

    def __build_parser(self) -> argparse.ArgumentParser:
        """
        Build root parser and command subparsers.
        """

        parser = argparse.ArgumentParser(
            description="Fathom: AI-powered mobile automation agent",
            allow_abbrev=False,
        )
        subparsers = parser.add_subparsers(dest="command", help="Command to execute")
        self.__configure_run_parser(subparsers=subparsers)
        self.__configure_explore_parser(subparsers=subparsers)
        return parser

    def __configure_run_parser(
        self,
        *,
        subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    ) -> None:
        """
        Configure run command arguments.
        """

        run_parser = subparsers.add_parser("run", help="Run an intent-based automation workflow")
        run_parser.add_argument("intent", type=str, help="The goal description")
        run_parser.add_argument(
            "--platform",
            type=str,
            default=None,
            choices=["android", "ios"],
            help="Local device platform",
        )
        run_parser.add_argument("--serial", "-s", type=str, help="Device serial number")
        run_parser.add_argument(
            "--ios-device-identifier",
            type=str,
            dest="ios_device_identifier",
            help="iOS simulator device identifier",
        )
        run_parser.add_argument(
            "--ios-bundle-identifier",
            "--bundle-id",
            type=str,
            dest="ios_bundle_identifier",
            help="Default iOS bundle identifier context",
        )
        run_parser.add_argument(
            "--ios-executable-path",
            type=str,
            help="Path to xcrun executable for iOS local adapter",
        )
        run_parser.add_argument(
            "--ios-automation-backend",
            type=str,
            default=None,
            choices=["xcrun_simctl", "xcuitest", "webdriver_agent"],
            help="iOS perception strategy: native xcrun or enhanced hierarchy via XCUITest/WebDriverAgent",
        )
        run_parser.add_argument(
            "--ios-web-driver-agent-url",
            type=str,
            default=None,
            help="WebDriverAgent URL for iOS hierarchy extraction",
        )
        run_parser.add_argument(
            "--ios-web-driver-agent-bundle-identifier",
            type=str,
            default=None,
            help="Bundle identifier used while creating WebDriverAgent sessions",
        )
        run_parser.add_argument(
            "--ios-web-driver-agent-request-timeout-seconds",
            type=float,
            default=None,
            help="WebDriverAgent request timeout in seconds",
        )
        run_parser.add_argument(
            "--adb-path",
            type=str,
            default=None,
            help="Path to adb executable for Android local adapter",
        )
        run_parser.add_argument("--api-key", "-k", type=str, help="Gemini API Key")
        run_parser.add_argument("--max-steps", type=int, default=50, help="Maximum steps allowed")
        run_parser.add_argument(
            "--verbose",
            "-v",
            action="store_true",
            help="Enable verbose output",
        )
        run_parser.add_argument(
            "--log-file",
            nargs="?",
            const="auto",
            default=None,
            dest="log_file",
            help=(
                "Tee structured logs to a file. Pass without a value to auto-resolve "
                "logs/<DD-MM-YYYY>/<workflow_id>/run.log, or pass an explicit path."
            ),
        )
        run_parser.add_argument(
            "--use-xml",
            "-x",
            action="store_true",
            help="Use XML bounding boxes",
        )
        run_parser.add_argument(
            "--interactive",
            "-i",
            action="store_true",
            help="Enable interactive HITL mode",
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
            help="Maximum allowed consecutive HITL realignments",
        )
        run_parser.add_argument(
            "--no-realignment",
            action="store_false",
            dest="immediate_realignment",
            help="Disable immediate realignment on context injection",
        )
        run_parser.set_defaults(immediate_realignment=True)

    def __configure_explore_parser(
        self,
        *,
        subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    ) -> None:
        """
        Configure explore command arguments.
        """

        explore_parser = subparsers.add_parser("explore", help="Run app exploration")
        explore_parser.add_argument(
            "--max-steps",
            type=int,
            default=50,
            help="Maximum steps allowed",
        )
        explore_parser.add_argument(
            "--package",
            type=str,
            default=None,
            dest="package_name",
            help="Application package/bundle identifier to launch and explore",
        )
        explore_parser.add_argument(
            "--focus",
            type=str,
            default=None,
            dest="focus",
            help="Steer exploration toward a specific flow or feature",
        )
        explore_parser.add_argument(
            "--platform",
            type=str,
            default=None,
            choices=["android", "ios"],
            help="Local device platform",
        )
        explore_parser.add_argument("--serial", "-s", type=str, help="Device serial number")
        explore_parser.add_argument(
            "--ios-device-identifier",
            type=str,
            dest="ios_device_identifier",
            help="iOS simulator device identifier",
        )
        explore_parser.add_argument(
            "--ios-bundle-identifier",
            "--bundle-id",
            type=str,
            dest="ios_bundle_identifier",
            help="Default iOS bundle identifier context",
        )
        explore_parser.add_argument(
            "--ios-executable-path",
            type=str,
            help="Path to xcrun executable for iOS local adapter",
        )
        explore_parser.add_argument(
            "--ios-automation-backend",
            type=str,
            default=None,
            choices=["xcrun_simctl", "xcuitest", "webdriver_agent"],
            help="iOS perception strategy: native xcrun or enhanced hierarchy via XCUITest/WebDriverAgent",
        )
        explore_parser.add_argument(
            "--ios-web-driver-agent-url",
            type=str,
            default=None,
            help="WebDriverAgent URL for iOS hierarchy extraction",
        )
        explore_parser.add_argument(
            "--ios-web-driver-agent-bundle-identifier",
            type=str,
            default=None,
            help="Bundle identifier used while creating WebDriverAgent sessions",
        )
        explore_parser.add_argument(
            "--ios-web-driver-agent-request-timeout-seconds",
            type=float,
            default=None,
            help="WebDriverAgent request timeout in seconds",
        )
        explore_parser.add_argument(
            "--adb-path",
            type=str,
            default=None,
            help="Path to adb executable for Android local adapter",
        )
        explore_parser.add_argument(
            "--verbose",
            "-v",
            action="store_true",
            help="Enable verbose output",
        )
        explore_parser.add_argument(
            "--tui",
            action="store_true",
            help="Render the run in a live full-screen exploration UI",
        )

    def __build_device_configuration(
        self,
        *,
        settings: FathomSettings,
        command_input: LocalCommandInput,
    ) -> DeviceConfiguration:
        """
        Resolve local device configuration from command input.
        """

        return self.__local_device_resolver.resolve(
            settings=settings,
            command_input=command_input,
        )

    def __build_run_request(
        self,
        *,
        settings: FathomSettings,
        command_input: RunCommandInput,
    ) -> IntentRunRequest:
        """
        Build canonical run request for the run command.
        """

        realignment = RealignmentPolicy(
            budget=command_input.realignment_budget,
            immediate=command_input.immediate_realignment,
        )
        device_configuration = self.__build_device_configuration(
            settings=settings,
            command_input=command_input,
        )

        runtime_configuration = RuntimeConfiguration(
            interactive=command_input.interactive,
            signal_type=SignalAdapterType(command_input.signal),
        )
        self.__activate_log_file_if_requested(
            settings=settings,
            log_file=command_input.log_file,
            workflow_id=runtime_configuration.session_id,
        )

        return IntentRunRequest(
            runtime=runtime_configuration,
            objective=IntentObjectiveConfiguration(
                intent=command_input.intent,
                use_xml=command_input.use_xml,
                max_steps=command_input.max_steps,
            ),
            memory=MemoryConfiguration(),
            resources=ResourceConfiguration(
                targets=[
                    TargetConfiguration(
                        kind=TargetKind.DEVICE,
                        device_configuration=device_configuration,
                    )
                ],
                language_model_configuration=ModelSelectionConfiguration(),
            ),
            metadata=RunMetadata(),
            interaction=InteractionConfiguration(realignment=realignment),
        )

    def __build_explore_request(
        self,
        *,
        settings: FathomSettings,
        command_input: ExploreCommandInput,
    ) -> ExplorationRunRequest:
        """
        Build canonical run request for the explore command.
        """

        device_configuration = self.__build_device_configuration(
            settings=settings,
            command_input=command_input,
        )

        return ExplorationRunRequest(
            objective=ExplorationObjectiveConfiguration(
                max_steps=command_input.max_steps,
                package_name=command_input.package_name,
                intent=command_input.focus or DEFAULT_EXPLORATION_INTENT,
                focus=command_input.focus,
            ),
            runtime=RuntimeConfiguration(
                interactive=False,
                signal_type=SignalAdapterType.INTERACTIVE,
            ),
            memory=MemoryConfiguration(),
            resources=ResourceConfiguration(
                targets=[
                    TargetConfiguration(
                        kind=TargetKind.DEVICE,
                        device_configuration=device_configuration,
                    )
                ],
                language_model_configuration=ModelSelectionConfiguration(),
            ),
            metadata=RunMetadata(),
        )

    def __resolve_command_inputs(
        self,
        *,
        settings: FathomSettings,
        arguments: Dict[str, object],
    ) -> Tuple[str, Optional[RunCommandInput], Optional[ExploreCommandInput]]:
        """
        Validate and resolve command-specific inputs.
        """

        command_name = str(arguments.get("command") or "")
        run_command_input: Optional[RunCommandInput] = None
        explore_command_input: Optional[ExploreCommandInput] = None

        if command_name == "run":
            run_command_input = RunCommandInput.model_validate(arguments)
            self.__apply_run_overrides(settings=settings, command_input=run_command_input)

        elif command_name == "explore":
            explore_command_input = ExploreCommandInput.model_validate(arguments)
            self.__apply_explore_overrides(settings=settings, command_input=explore_command_input)

        return command_name, run_command_input, explore_command_input

    def __apply_run_overrides(
        self,
        *,
        settings: FathomSettings,
        command_input: RunCommandInput,
    ) -> None:
        """
        Apply validated settings overrides for run command.
        """

        if command_input.api_key:
            settings.gemini_api_key = command_input.api_key

        if command_input.verbose:
            settings.log_level = "DEBUG"

        if command_input.adb_executable_path:
            settings.adb_path = command_input.adb_executable_path

    def __apply_explore_overrides(
        self,
        *,
        settings: FathomSettings,
        command_input: ExploreCommandInput,
    ) -> None:
        """
        Apply validated settings overrides for explore command.
        """

        if command_input.verbose:
            settings.log_level = "DEBUG"

        if command_input.adb_executable_path:
            settings.adb_path = command_input.adb_executable_path

    @staticmethod
    def __activate_log_file_if_requested(
        *,
        workflow_id: str,
        log_file: Optional[str],
        settings: FathomSettings,
    ) -> None:
        """
        Resolve the --log-file argument and tee the structured log stream to it.
        """

        if log_file is None:
            return

        if log_file == "auto":
            today = datetime.datetime.now().strftime("%d-%m-%Y")
            target = settings.run_logs_path / today / workflow_id / "run.log"
        else:
            target = Path(log_file).expanduser()

        target = target.resolve()
        BaseLogger.attach_file_handler(path=target)

        console.print(f"[dim]Logs teeing to:[/dim] [cyan]{target}[/cyan]")

    def __configure_logging(self, *, settings: FathomSettings) -> None:
        """
        Configure logging for command execution.
        """

        BaseLogger.configure(settings)
        getLogger("PIL").setLevel("ERROR")
        getLogger("urllib3").setLevel("ERROR")
        getLogger("httpx").setLevel("WARNING")
        getLogger("google").setLevel("WARNING")
        getLogger("google.auth").setLevel("WARNING")

    def __dispatch(
        self,
        *,
        command_name: str,
        settings: FathomSettings,
        run_command_input: Optional[RunCommandInput],
        explore_command_input: Optional[ExploreCommandInput],
    ) -> int:
        """
        Dispatch validated command to runtime executor.
        """

        executor = CommandExecutor(settings=settings)

        if command_name == "run":
            if run_command_input is None:
                console.print("[bold red]Invalid run command input.[/bold red]")
                return 1

            run_request = self.__build_run_request(
                settings=settings,
                command_input=run_command_input,
            )
            return asyncio.run(executor.run(request=run_request))

        if command_name == "explore":
            if explore_command_input is None:
                console.print("[bold red]Invalid explore command input.[/bold red]")
                return 1

            exploration_request = self.__build_explore_request(
                settings=settings,
                command_input=explore_command_input,
            )
            if explore_command_input.tui:
                return self.__run_exploration_tui(executor=executor, request=exploration_request)
            return asyncio.run(executor.explore(request=exploration_request))

        self.__parser.print_help()
        return 0

    @staticmethod
    def __run_exploration_tui(*, executor: CommandExecutor, request: ExplorationRunRequest) -> int:
        """
        Launch the full-screen explorer UI, driving the run on its worker thread.

        Console logging is detached first so engine logs cannot corrupt the
        Textual display; the activity log surfaces the run instead.
        """

        from fathom.runtime.command.explorer_tui import ExplorerApp

        BaseLogger.silence_console()

        async def workflow() -> bool:
            return await executor.explore(request=request, view=app) == 0

        app = ExplorerApp(
            package=request.objective.package_name or "",
            max_steps=request.objective.max_steps,
            workflow=workflow,
            on_cancel=executor.cancel,
        )
        app.run()
        return app.exit_code

    def run(self) -> int:
        """
        Execute command application lifecycle.
        """

        arguments = vars(self.__parser.parse_args())
        settings = FathomSettings()

        try:
            command_name, run_command_input, explore_command_input = self.__resolve_command_inputs(
                arguments=arguments,
                settings=settings,
            )
        except ValidationError as exception:
            console.print(f"[bold red]Invalid CLI arguments:[/bold red] {escape(str(exception))}")
            return 1

        self.__configure_logging(settings=settings)

        try:
            return self.__dispatch(
                command_name=command_name,
                settings=settings,
                run_command_input=run_command_input,
                explore_command_input=explore_command_input,
            )
        except FathomError as exception:
            console.print(f"[bold red]Configuration Error:[/bold red] {escape(str(exception))}")
            return 1
        except KeyboardInterrupt:
            return 1
