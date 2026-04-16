from __future__ import annotations

import argparse
import asyncio
from logging import getLogger
from typing import Dict, Optional, Tuple

from pydantic import ValidationError
from rich.console import Console
from rich.markup import escape

from fathom.base.logger import BaseLogger
from fathom.constants.run import SignalAdapterType, TargetKind
from fathom.core.exceptions import FathomError
from fathom.runtime.bootstrap import register_default_prompt_builders
from fathom.runtime.command.executor import CommandExecutor
from fathom.runtime.command.resolver import (
    LocalDeviceConfigurationResolver,
    LocalDeviceConfigurationResolverPort,
)
from fathom.runtime.command.wizard import InteractiveWizard, wizard_argv
from fathom.schemas.cli import (
    DemoCommandInput,
    LocalCommandInput,
    RunCommandInput,
)
from fathom.schemas.configuration import DeviceConfiguration
from fathom.schemas.run import (
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

        Also wires the provider registries that core services depend on
        (prompt builders, etc.) so that instantiating a service inside a
        CLI command does not rely on module import side effects.
        """

        register_default_prompt_builders()

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
        self.__configure_demo_parser(subparsers=subparsers)
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
        run_parser.add_argument("--max-steps", type=int, default=100, help="Maximum steps allowed")
        run_parser.add_argument(
            "--verbose",
            "-v",
            action="store_true",
            help="Enable verbose output",
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
            help="Maximum allowed consecutive re-plans",
        )
        run_parser.add_argument(
            "--no-realignment",
            action="store_false",
            dest="immediate_realignment",
            help="Disable immediate re-planning on context injection",
        )
        run_parser.set_defaults(immediate_realignment=True)

    def __configure_demo_parser(
        self,
        *,
        subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    ) -> None:
        """
        Configure demo command arguments.

        The ``demo`` command is a styled variant of ``run`` that swaps
        in the live footer telemetry adapter for Claude-Code-style
        presentation. All flags mirror ``run``; defaults are tuned for
        showcases (XML grounding on, verbose off).
        """

        demo_parser = subparsers.add_parser(
            "demo",
            help="Run an intent workflow with a demo-styled live footer",
        )
        demo_parser.add_argument("intent", type=str, help="The goal description")
        demo_parser.add_argument(
            "--platform",
            type=str,
            default=None,
            choices=["android", "ios"],
            help="Local device platform",
        )
        demo_parser.add_argument("--serial", "-s", type=str, help="Device serial number")
        demo_parser.add_argument(
            "--ios-device-identifier",
            type=str,
            dest="ios_device_identifier",
            help="iOS simulator device identifier",
        )
        demo_parser.add_argument(
            "--ios-bundle-identifier",
            "--bundle-id",
            type=str,
            dest="ios_bundle_identifier",
            help="Default iOS bundle identifier context",
        )
        demo_parser.add_argument(
            "--ios-executable-path",
            type=str,
            help="Path to xcrun executable for iOS local adapter",
        )
        demo_parser.add_argument(
            "--ios-automation-backend",
            type=str,
            default=None,
            choices=["xcrun_simctl", "xcuitest", "webdriver_agent"],
            help="iOS perception strategy",
        )
        demo_parser.add_argument(
            "--ios-web-driver-agent-url",
            type=str,
            default=None,
            help="WebDriverAgent URL for iOS hierarchy extraction",
        )
        demo_parser.add_argument(
            "--ios-web-driver-agent-bundle-identifier",
            type=str,
            default=None,
            help="Bundle identifier used while creating WebDriverAgent sessions",
        )
        demo_parser.add_argument(
            "--ios-web-driver-agent-request-timeout-seconds",
            type=float,
            default=None,
            help="WebDriverAgent request timeout in seconds",
        )
        demo_parser.add_argument(
            "--adb-path",
            type=str,
            default=None,
            help="Path to adb executable for Android local adapter",
        )
        demo_parser.add_argument("--api-key", "-k", type=str, help="Gemini API Key")
        demo_parser.add_argument("--max-steps", type=int, default=100, help="Maximum steps allowed")
        demo_parser.add_argument(
            "--verbose",
            "-v",
            action="store_true",
            help="Enable verbose output",
        )
        demo_parser.add_argument(
            "--realignment-budget",
            type=int,
            default=3,
            help="Maximum allowed consecutive re-plans",
        )
        demo_parser.add_argument(
            "--no-realignment",
            action="store_false",
            dest="immediate_realignment",
            help="Disable immediate re-planning on context injection",
        )
        # Demo is the interactive HITL variant of run:
        # - ``interactive=True`` by default so the signal adapter is
        #   wired and the agent escalates to a human prompt on
        #   uncertainty.
        # - ``signal="interactive"`` picks the interactive HITL adapter
        #   (console prompts + stdin).
        # - ``use_xml`` intentionally omitted so it defaults to False
        #   (matching ``run``).
        # The non-HITL autonomous path is ``fathom run``.
        demo_parser.set_defaults(
            immediate_realignment=True,
            interactive=True,
            signal="interactive",
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

        return IntentRunRequest(
            objective=IntentObjectiveConfiguration(
                intent=command_input.intent,
                use_xml=command_input.use_xml,
                max_steps=command_input.max_steps,
            ),
            runtime=RuntimeConfiguration(
                interactive=command_input.interactive,
                signal_type=SignalAdapterType(command_input.signal),
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
            interaction=InteractionConfiguration(realignment=realignment),
            metadata=RunMetadata(),
        )

    def __resolve_command_inputs(
        self,
        *,
        settings: FathomSettings,
        arguments: Dict[str, object],
    ) -> Tuple[str, Optional[RunCommandInput]]:
        """
        Validate and resolve command-specific inputs.
        """

        command_name = str(arguments.get("command") or "")
        run_command_input: Optional[RunCommandInput] = None

        if command_name == "run":
            run_command_input = RunCommandInput.model_validate(arguments)
            self.__apply_run_overrides(settings=settings, command_input=run_command_input)

        elif command_name == "demo":
            # DemoCommandInput extends RunCommandInput; routing the
            # parsed demo args through it lets the downstream flow
            # (build_run_request, executor) treat demo as a styled
            # variant of run.
            run_command_input = DemoCommandInput.model_validate(arguments)
            self.__apply_run_overrides(settings=settings, command_input=run_command_input)

        return command_name, run_command_input

    def __apply_run_overrides(
        self,
        *,
        settings: FathomSettings,
        command_input: RunCommandInput,
    ) -> None:
        """
        Apply validated settings overrides for run / demo commands.

        Both commands default to WARNING log level so per-step INFO
        and DEBUG lines don't bury the rendered panels / live footer.
        ``--verbose`` opts back into DEBUG for full operator detail.
        Rich panels (``AuditService``, banner, summary tables, demo
        footer) are ``console.print`` / ``rich.live`` renderables and
        are not affected by log level — only structlog text output is.
        """

        if command_input.api_key:
            settings.gemini_api_key = command_input.api_key

        if command_input.verbose:
            settings.log_level = "DEBUG"
        else:
            settings.log_level = "WARNING"

        if command_input.adb_executable_path:
            settings.adb_path = command_input.adb_executable_path

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
    ) -> int:
        """
        Dispatch validated command to runtime executor.
        """

        executor = CommandExecutor(settings=settings, demo_mode=(command_name == "demo"))

        if command_name in {"run", "demo"}:
            if run_command_input is None:
                console.print(f"[bold red]Invalid {command_name} command input.[/bold red]")
                return 1

            run_request = self.__build_run_request(
                settings=settings,
                command_input=run_command_input,
            )
            return asyncio.run(executor.run(request=run_request))

        self.__parser.print_help()
        return 0

    def run(self) -> int:
        """
        Execute command application lifecycle.

        When invoked without a subcommand (typical for ``Drizz`` /
        ``fathom`` bare), the interactive wizard collects inputs and
        we re-enter this flow with a synthesized argv so argparse +
        Pydantic validation stay the single source of truth.
        """

        parsed_args = self.__parser.parse_args()
        arguments = vars(parsed_args)

        if not arguments.get("command"):
            return self.__run_wizard_flow()

        return self.__run_with_arguments(arguments=arguments)

    def __run_wizard_flow(self) -> int:
        """
        Launch the interactive wizard and re-dispatch its result.
        """

        try:
            wizard_result = InteractiveWizard(console=console).run()
        except KeyboardInterrupt:
            return 1

        if wizard_result is None:
            console.print("[yellow]Aborted.[/yellow]")
            return 0

        argv = wizard_argv(args=wizard_result)
        arguments = vars(self.__parser.parse_args(argv))
        return self.__run_with_arguments(arguments=arguments)

    def __run_with_arguments(self, *, arguments: Dict[str, object]) -> int:
        """
        Resolve inputs and dispatch to the appropriate command.
        """

        settings = FathomSettings()

        try:
            command_name, run_command_input = self.__resolve_command_inputs(
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
            )
        except FathomError as exception:
            console.print(f"[bold red]Configuration Error:[/bold red] {escape(str(exception))}")
            return 1
        except KeyboardInterrupt:
            return 1
