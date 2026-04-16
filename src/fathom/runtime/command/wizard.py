"""
Interactive launcher for the ``Drizz`` / ``fathom`` shell command.

When the user types ``Drizz`` (or ``fathom``) with no subcommand, the
CLI dispatches to ``InteractiveWizard.run()`` which walks through a
short ``rich``-styled prompt flow collecting the command type, intent,
and platform / device flags, then returns an argparse-shaped dict for
the main dispatcher to validate and execute. No bespoke validation —
the wizard only collects; Pydantic validators in ``schemas/cli.py``
still enforce correctness.
"""

from __future__ import annotations

import json
import shutil
import subprocess  # nosec - used with explicit shell=False + fixed argv
from logging import getLogger
from typing import Any, Dict, List, Optional, Sequence, cast

import questionary
from questionary import Style
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.table import Table
from rich.text import Text

logger = getLogger(__name__)

_COMMAND_CHOICES = ["demo", "run"]
_PLATFORM_CHOICES = ["android", "ios"]
_IOS_BACKEND_CHOICES = ["xcuitest", "webdriver_agent", "xcrun_simctl"]
_DEFAULT_MAX_STEPS = 100
_DEFAULT_COMMAND = "demo"
_DEFAULT_PLATFORM = "android"
_DEFAULT_IOS_BACKEND = "xcuitest"

# Style tokens used by every questionary.select menu so they read as
# one coherent palette with the rest of the Drizz wordmark + splash.
_QUESTIONARY_STYLE = Style(
    [
        ("qmark", "fg:#a88fd8 bold"),
        ("question", "fg:#a88fd8 bold"),
        ("pointer", "fg:#a88fd8 bold"),
        ("highlighted", "fg:#6b3fd4 bold"),
        ("selected", "fg:#6b3fd4 bold"),
        ("answer", "fg:#a88fd8 bold"),
    ]
)

# "DRIZZ" in 5-row block-letter art. Each letter lives in a 5-col
# grid; letters are joined with a single-space separator → 29 cols
# total. Strings are composed at import time so hand-counting spaces
# never drifts.
_LETTER_GLYPHS = {
    # D with "curved" top-right and bottom-right corners (rows 0 and 4
    # pulled in by one col), so it reads as a D rather than a
    # rectangle. Middle rows extend to col 5 to form the bulge.
    "D": ("████ ", "█   █", "█   █", "█   █", "████ "),
    "R": ("█████", "█   █", "█████", "█  █ ", "█   █"),
    "I": ("█████", "  █  ", "  █  ", "  █  ", "█████"),
    "Z": ("█████", "   █ ", "  █  ", " █   ", "█████"),
}

_WORDMARK_LINES = tuple(
    "[#a88fd8]" + " ".join(_LETTER_GLYPHS[letter][row] for letter in "DRIZZ") + "[/]"
    for row in range(5)
)


class InteractiveWizard:
    """
    Prompt-based Fathom launcher used when ``Drizz`` is invoked with no
    subcommand.
    """

    def __init__(self, *, console: Optional[Console] = None) -> None:
        """
        Initialize the wizard with an optional Console override.
        """

        self.__console = console or Console()

    # ------------------------------------------------------------------
    # Public entry
    # ------------------------------------------------------------------

    def run(self) -> Optional[Dict[str, Any]]:
        """
        Collect inputs interactively and return an argparse-shaped dict.

        Returns ``None`` when the user declines the final confirmation.
        Any exception raised by ``rich.prompt`` (e.g. Ctrl-C) is left
        to propagate so the outer CLI can handle it uniformly with the
        non-wizard path.
        """

        self.__print_banner()

        command = self.__prompt_command()

        platform = self.__prompt_platform()
        device_args = self.__prompt_device(platform=platform)
        ios_args = self.__prompt_ios_details(platform=platform) if platform == "ios" else {}
        max_steps = self.__prompt_max_steps()
        verbose = self.__prompt_verbose()

        intent = self.__prompt_intent()

        args: Dict[str, Any] = {
            "command": command,
            "intent": intent,
            "platform": platform,
            "max_steps": max_steps,
            "verbose": verbose,
            **device_args,
            **ios_args,
        }

        if not self.__confirm(args=args):
            return None

        return args

    # ------------------------------------------------------------------
    # Individual prompts
    # ------------------------------------------------------------------

    def __print_banner(self) -> None:
        """
        Render the DRIZZ wordmark followed by the splash panel.
        """

        self.__console.print()
        for line in _WORDMARK_LINES:
            self.__console.print(line)
        self.__console.print()
        self.__console.print(
            Panel.fit(
                "[bold #a88fd8]Drizz[/bold #a88fd8] [dim]·[/dim] "
                "interactive launcher for Fathom\n"
                "[dim]Answer the prompts, confirm, and the agent runs.[/dim]",
                border_style="#6b3fd4",
            )
        )

    def __prompt_select(
        self,
        *,
        title: str,
        options: Sequence[str],
        default_index: int = 0,
    ) -> str:
        """
        Render an arrow-key select menu and return the selected option.

        Uses ``questionary.select`` so the user navigates with ↑/↓ and
        confirms with Enter. Typed-filter is enabled by the library —
        pressing a letter jumps to the next option starting with it.

        Ctrl-C / Ctrl-D at the menu returns ``None`` from questionary,
        which we translate into ``KeyboardInterrupt`` so the wizard's
        outer handler can cancel uniformly with the rest of the flow.
        """

        if not options:
            raise ValueError("options must be non-empty")

        clamped_default = max(0, min(default_index, len(options) - 1))

        # questionary uses a single plain-text prompt string; strip
        # rich markup so the menu title reads cleanly.
        plain_title = Text.from_markup(title).plain

        selected = questionary.select(
            plain_title,
            choices=list(options),
            default=options[clamped_default],
            style=_QUESTIONARY_STYLE,
            use_shortcuts=False,
            instruction="(↑/↓ to move, Enter to select)",
        ).ask()

        if selected is None:
            raise KeyboardInterrupt("Selection cancelled")

        return cast("str", selected)

    def __prompt_command(self) -> str:
        """
        Pick the Fathom subcommand via arrow-key menu.
        """

        return self.__prompt_select(
            title="Command",
            options=_COMMAND_CHOICES,
            default_index=_COMMAND_CHOICES.index(_DEFAULT_COMMAND),
        )

    def __prompt_intent(self) -> str:
        """
        Collect the goal description from the user.
        """

        while True:
            raw = cast(
                "str",
                Prompt.ask(
                    "[cyan]What are we testing today?[/cyan]",
                    console=self.__console,
                ),
            )
            intent = raw.strip()
            if intent:
                return intent
            self.__console.print("[yellow]Intent cannot be empty.[/yellow]")

    def __prompt_platform(self) -> str:
        """
        Pick the target device platform via arrow-key menu.
        """

        return self.__prompt_select(
            title="Platform",
            options=_PLATFORM_CHOICES,
            default_index=_PLATFORM_CHOICES.index(_DEFAULT_PLATFORM),
        )

    def __prompt_device(self, *, platform: str) -> Dict[str, Any]:
        """
        Pick the device identifier via numbered menu of detected devices
        with "enter manually" / "skip" fallback options. When nothing is
        auto-detected, falls back to a single free-text prompt.
        """

        detected = self.__detect_devices(platform=platform)
        label = "iOS device identifier" if platform == "ios" else "Android serial"
        key = "ios_device_identifier" if platform == "ios" else "serial"

        if detected:
            options = [*detected, "enter manually", "skip (use default)"]
            choice = self.__prompt_select(
                title=f"Select {label}",
                options=options,
                default_index=0,
            )

            if choice == "enter manually":
                manual = cast(
                    "str",
                    Prompt.ask(
                        f"[cyan]{label}[/cyan]",
                        default="",
                        console=self.__console,
                    ),
                ).strip()
                return {key: manual} if manual else {}

            if choice == "skip (use default)":
                return {}

            return {key: choice}

        # No devices detected: free-text fallback.
        raw = cast(
            "str",
            Prompt.ask(
                f"[cyan]{label}[/cyan] [dim](blank = use default)[/dim]",
                default="",
                console=self.__console,
            ),
        ).strip()
        return {key: raw} if raw else {}

    def __prompt_ios_details(self, *, platform: str) -> Dict[str, Any]:
        """
        Collect iOS-specific flags.
        """

        if platform != "ios":
            return {}

        bundle = cast(
            "str",
            Prompt.ask(
                "[cyan]iOS bundle identifier[/cyan] [dim](blank to skip)[/dim]",
                default="",
                console=self.__console,
            ),
        ).strip()

        backend = self.__prompt_select(
            title="iOS automation backend",
            options=_IOS_BACKEND_CHOICES,
            default_index=_IOS_BACKEND_CHOICES.index(_DEFAULT_IOS_BACKEND),
        )

        args: Dict[str, Any] = {"ios_automation_backend": backend}
        if bundle:
            args["ios_bundle_identifier"] = bundle
        return args

    def __prompt_max_steps(self) -> int:
        """
        Collect the max step budget.
        """

        return cast(
            "int",
            IntPrompt.ask(
                "[cyan]Max steps[/cyan]",
                default=_DEFAULT_MAX_STEPS,
                console=self.__console,
            ),
        )

    def __prompt_verbose(self) -> bool:
        """
        Opt into verbose (DEBUG-level) logging.
        """

        return cast(
            "bool",
            Confirm.ask(
                "[cyan]Verbose logs?[/cyan] [dim](shows structlog DEBUG/INFO)[/dim]",
                default=False,
                console=self.__console,
            ),
        )

    def __confirm(self, *, args: Dict[str, Any]) -> bool:
        """
        Render a summary panel and ask the user to confirm.
        """

        summary = Table.grid(padding=(0, 2))
        summary.add_column(style="bold cyan", justify="right")
        summary.add_column(overflow="ellipsis")

        summary.add_row("command", str(args["command"]))
        if args.get("intent"):
            intent_display = str(args["intent"])
            if len(intent_display) > 200:
                intent_display = intent_display[:200] + "…"
            summary.add_row("intent", intent_display)
        summary.add_row("platform", str(args["platform"]))

        for key in (
            "serial",
            "ios_device_identifier",
            "ios_bundle_identifier",
            "ios_automation_backend",
        ):
            if args.get(key):
                summary.add_row(key, str(args[key]))

        summary.add_row("max_steps", str(args["max_steps"]))
        summary.add_row("verbose", str(args["verbose"]))

        self.__console.print(Panel(summary, title="Launch?", border_style="green", padding=(0, 1)))

        return cast(
            "bool",
            Confirm.ask(
                "[bold green]Proceed?[/bold green]",
                default=True,
                console=self.__console,
            ),
        )

    # ------------------------------------------------------------------
    # Auto-detection (best-effort; never raises)
    # ------------------------------------------------------------------

    def __detect_devices(self, *, platform: str) -> List[str]:
        """
        Return a list of detected device identifiers. Never raises.
        """

        try:
            if platform == "android":
                return self.__detect_adb_devices()
            if platform == "ios":
                return self.__detect_ios_simulators()
        except Exception as exception:
            logger.debug("Device auto-detection failed: %s", exception)
        return []

    def __detect_adb_devices(self) -> List[str]:
        """
        Parse ``adb devices`` output.
        """

        adb = shutil.which("adb")
        if not adb:
            return []

        result = subprocess.run(  # nosec - fixed argv, shell=False default
            [adb, "devices"],
            capture_output=True,
            text=True,
            timeout=1.5,
            check=False,
        )
        if result.returncode != 0:
            return []

        serials: List[str] = []
        for line in result.stdout.splitlines()[1:]:
            stripped = line.strip()
            if not stripped or "\t" not in stripped:
                continue
            serial, state = stripped.split("\t", 1)
            if state.strip() == "device":
                serials.append(serial.strip())
        return serials

    def __detect_ios_simulators(self) -> List[str]:
        """
        Parse ``xcrun simctl list devices booted -j`` output.
        """

        xcrun = shutil.which("xcrun")
        if not xcrun:
            return []

        result = subprocess.run(  # nosec - fixed argv, shell=False default
            [xcrun, "simctl", "list", "devices", "booted", "-j"],
            capture_output=True,
            text=True,
            timeout=1.5,
            check=False,
        )
        if result.returncode != 0:
            return []

        try:
            payload = json.loads(result.stdout)
        except ValueError:
            return []

        devices = payload.get("devices", {}) or {}
        udids: List[str] = []
        for runtime_devices in devices.values():
            if not isinstance(runtime_devices, list):
                continue
            for device in runtime_devices:
                if isinstance(device, dict) and device.get("state") == "Booted":
                    udid = device.get("udid")
                    if isinstance(udid, str) and udid:
                        udids.append(udid)
        return udids


def wizard_argv(args: Dict[str, Any]) -> List[str]:
    """
    Convert a wizard result dict into an argv list argparse can parse.

    Factored out of ``CommandApplication`` so it can be unit-tested in
    isolation and reused wherever the wizard result needs to be
    materialized back into CLI arguments.
    """

    if "command" not in args:
        raise ValueError("wizard result missing 'command' key")

    argv: List[str] = [str(args["command"])]

    # Both supported commands (run / demo) take a positional intent.
    if args.get("intent"):
        argv.append(str(args["intent"]))

    # Optional flags: emit only when truthy / non-empty.
    flag_map = (
        ("platform", "--platform"),
        ("serial", "--serial"),
        ("ios_device_identifier", "--ios-device-identifier"),
        ("ios_bundle_identifier", "--ios-bundle-identifier"),
        ("ios_automation_backend", "--ios-automation-backend"),
    )
    for key, flag in flag_map:
        value = args.get(key)
        if value:
            argv += [flag, str(value)]

    if args.get("max_steps") is not None:
        argv += ["--max-steps", str(args["max_steps"])]

    if args.get("verbose"):
        argv.append("--verbose")

    return argv
