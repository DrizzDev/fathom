"""
Interactive launcher for the ``fathom`` exploration command.

When the user types ``fathom`` with no subcommand, the CLI dispatches
to ``InteractiveWizard.run()`` which walks through a short
``rich``-styled prompt flow collecting the device, target package,
focus, and budget, then returns an argparse-shaped dict for the main
dispatcher to validate and execute. The wizard only collects; argparse
defaults still enforce correctness on re-parse.
"""

from __future__ import annotations

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

_DEFAULT_MAX_STEPS = 50
_MAX_PACKAGE_CHOICES = 25

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

_LETTER_GLYPHS = {
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
    Prompt-based Fathom Explorer launcher used when ``fathom`` is invoked
    with no subcommand.
    """

    def __init__(self, *, console: Optional[Console] = None) -> None:
        self.__console = console or Console()

    def run(self) -> Optional[Dict[str, Any]]:
        """
        Collect inputs interactively and return an argparse-shaped dict.

        Returns ``None`` when the user declines the final confirmation.
        """

        self.__print_banner()

        device_args = self.__prompt_device()
        package_args = self.__prompt_package(serial=device_args.get("serial"))
        focus_args = self.__prompt_focus()
        max_steps = self.__prompt_max_steps()
        verbose = self.__prompt_verbose()
        tui = self.__prompt_tui()

        args: Dict[str, Any] = {
            "command": "explore",
            "max_steps": max_steps,
            "verbose": verbose,
            "tui": tui,
            **device_args,
            **package_args,
            **focus_args,
        }

        if not self.__confirm(args=args):
            return None

        return args

    # ------------------------------------------------------------------
    # Individual prompts
    # ------------------------------------------------------------------

    def __print_banner(self) -> None:
        self.__console.print()
        for line in _WORDMARK_LINES:
            self.__console.print(line)
        self.__console.print()
        self.__console.print(
            Panel.fit(
                "[bold #a88fd8]Drizz[/bold #a88fd8] [dim]·[/dim] "
                "interactive launcher for Fathom Explorer\n"
                "[dim]Answer the prompts, confirm, and exploration begins.[/dim]",
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
        """

        if not options:
            raise ValueError("options must be non-empty")

        clamped_default = max(0, min(default_index, len(options) - 1))
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

    def __prompt_device(self) -> Dict[str, Any]:
        """
        Pick the Android device serial via numbered menu of detected
        devices with manual / skip fallbacks.
        """

        detected = self.__detect_adb_devices()
        label = "Android serial"

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
                return {"serial": manual} if manual else {}

            if choice == "skip (use default)":
                return {}

            return {"serial": choice}

        raw = cast(
            "str",
            Prompt.ask(
                f"[cyan]{label}[/cyan] [dim](blank = use default)[/dim]",
                default="",
                console=self.__console,
            ),
        ).strip()
        return {"serial": raw} if raw else {}

    def __prompt_package(self, *, serial: Optional[str]) -> Dict[str, Any]:
        """
        Pick the target package. Offers the foreground app + a slice of
        installed third-party packages + manual / skip fallbacks.
        """

        foreground = self.__detect_foreground_package(serial=serial)
        installed = self.__detect_installed_packages(serial=serial)

        options: List[str] = []
        if foreground:
            options.append(f"auto-detect (foreground: {foreground})")
        options.extend(installed[:_MAX_PACKAGE_CHOICES])
        options.append("enter manually")
        options.append("skip (auto-detect at runtime)")

        if len(options) == 2 and not foreground:
            # No foreground, no installed — go straight to free-text.
            raw = cast(
                "str",
                Prompt.ask(
                    "[cyan]Target package[/cyan] [dim](blank = auto-detect)[/dim]",
                    default="",
                    console=self.__console,
                ),
            ).strip()
            return {"package": raw} if raw else {}

        choice = self.__prompt_select(
            title="Select target package",
            options=options,
            default_index=0,
        )

        if choice.startswith("auto-detect"):
            return {"package": foreground} if foreground else {}

        if choice == "enter manually":
            manual = cast(
                "str",
                Prompt.ask(
                    "[cyan]Target package[/cyan]",
                    default="",
                    console=self.__console,
                ),
            ).strip()
            return {"package": manual} if manual else {}

        if choice == "skip (auto-detect at runtime)":
            return {}

        return {"package": choice}

    def __prompt_focus(self) -> Dict[str, Any]:
        """
        Free-text focus hint. Empty input means full-breadth mapping.
        """

        raw = cast(
            "str",
            Prompt.ask(
                "[cyan]Focus[/cyan] [dim](e.g. 'the checkout flow' — blank = full-breadth)[/dim]",
                default="",
                console=self.__console,
            ),
        ).strip()
        return {"focus": raw} if raw else {}

    def __prompt_max_steps(self) -> int:
        return cast(
            "int",
            IntPrompt.ask(
                "[cyan]Max steps[/cyan]",
                default=_DEFAULT_MAX_STEPS,
                console=self.__console,
            ),
        )

    def __prompt_verbose(self) -> bool:
        return cast(
            "bool",
            Confirm.ask(
                "[cyan]Verbose logs?[/cyan] [dim](shows DEBUG/INFO)[/dim]",
                default=False,
                console=self.__console,
            ),
        )

    def __prompt_tui(self) -> bool:
        return cast(
            "bool",
            Confirm.ask(
                "[cyan]Use live TUI?[/cyan] [dim](scrollable header/body/footer)[/dim]",
                default=True,
                console=self.__console,
            ),
        )

    def __confirm(self, *, args: Dict[str, Any]) -> bool:
        summary = Table.grid(padding=(0, 2))
        summary.add_column(style="bold cyan", justify="right")
        summary.add_column(overflow="ellipsis")

        summary.add_row("command", str(args["command"]))
        for key in ("serial", "package"):
            if args.get(key):
                summary.add_row(key, str(args[key]))
        if args.get("focus"):
            focus_display = str(args["focus"])
            if len(focus_display) > 200:
                focus_display = focus_display[:200] + "…"
            summary.add_row("focus", focus_display)
        summary.add_row("max_steps", str(args["max_steps"]))
        summary.add_row("verbose", str(args["verbose"]))
        summary.add_row("tui", str(args.get("tui", False)))

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

    def __detect_adb_devices(self) -> List[str]:
        adb = shutil.which("adb")
        if not adb:
            return []

        try:
            result = subprocess.run(  # nosec - fixed argv, shell=False default
                [adb, "devices"],
                capture_output=True,
                text=True,
                timeout=1.5,
                check=False,
            )
        except Exception as exception:
            logger.debug("adb devices failed: %s", exception)
            return []

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

    def __detect_installed_packages(self, *, serial: Optional[str]) -> List[str]:
        adb = shutil.which("adb")
        if not adb:
            return []

        argv = [adb]
        if serial:
            argv += ["-s", serial]
        argv += ["shell", "pm", "list", "packages", "-3"]

        try:
            result = subprocess.run(  # nosec - fixed argv, shell=False default
                argv,
                capture_output=True,
                text=True,
                timeout=3.0,
                check=False,
            )
        except Exception as exception:
            logger.debug("pm list packages failed: %s", exception)
            return []

        if result.returncode != 0:
            return []

        packages: List[str] = []
        for line in result.stdout.splitlines():
            stripped = line.strip()
            if stripped.startswith("package:"):
                pkg = stripped[len("package:") :].strip()
                if pkg:
                    packages.append(pkg)
        return sorted(packages)

    def __detect_foreground_package(self, *, serial: Optional[str]) -> Optional[str]:
        adb = shutil.which("adb")
        if not adb:
            return None

        argv = [adb]
        if serial:
            argv += ["-s", serial]
        argv += ["shell", "dumpsys", "window", "windows"]

        try:
            result = subprocess.run(  # nosec - fixed argv, shell=False default
                argv,
                capture_output=True,
                text=True,
                timeout=3.0,
                check=False,
            )
        except Exception as exception:
            logger.debug("dumpsys window failed: %s", exception)
            return None

        if result.returncode != 0:
            return None

        for line in result.stdout.splitlines():
            stripped = line.strip()
            if "mCurrentFocus" not in stripped and "mResumedActivity" not in stripped:
                continue
            for token in stripped.replace("{", " ").replace("}", " ").split():
                if "/" in token and "." in token.split("/", 1)[0]:
                    return token.split("/", 1)[0]
        return None


def wizard_argv(args: Dict[str, Any]) -> List[str]:
    """
    Convert a wizard result dict into an argv list argparse can parse.
    """

    if "command" not in args:
        raise ValueError("wizard result missing 'command' key")

    argv: List[str] = [str(args["command"])]

    flag_map = (
        ("package", "--package"),
        ("serial", "--serial"),
        ("focus", "--focus"),
    )
    for key, flag in flag_map:
        value = args.get(key)
        if value:
            argv += [flag, str(value)]

    if args.get("max_steps") is not None:
        argv += ["--max-steps", str(args["max_steps"])]

    if args.get("verbose"):
        argv.append("--verbose")

    if args.get("tui"):
        argv.append("--tui")

    return argv
