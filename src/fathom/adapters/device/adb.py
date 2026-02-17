from __future__ import annotations

import asyncio
import re
from logging import getLogger
from typing import List, Optional, Tuple

from rich.console import Console

from fathom.interfaces.device import DevicePort
from fathom.schemas.configuration import ADBConfig
from fathom.schemas.results import ActionResult

console = Console()
logger = getLogger(__name__)


class ADBDevice(DevicePort):
    """
    ADB adapter for Android devices.

    This adapter wraps the existing ADB tool logic without modifications.
    All methods are copied from tools/device/adb.py to preserve exact behavior.
    """

    def __init__(
        self, *, serial: Optional[str] = None, configuration: Optional[ADBConfig] = None
    ) -> None:
        """
        Initialize ADB device adapter.
        """

        if configuration:
            self.__configuration = configuration
        else:
            self.__configuration = ADBConfig(device_serial=serial) if serial else ADBConfig()

        self.__cached_size: Optional[Tuple[int, int]] = None

    @property
    def configuration(self) -> ADBConfig:
        """
        Returns the tool configuration.
        """

        return self.__configuration

    async def tap(self, *, x: int, y: int) -> ActionResult:
        """
        Execute tap at coordinates.
        """

        console.print(f"[bold cyan]🖱️  TAP[/bold cyan] at ([bold yellow]{x}, {y}[/bold yellow])")
        return await self.__shell(command=f"input tap {x} {y}")

    async def type_text(self, *, text: str) -> ActionResult:
        """
        Type text on device.
        """

        display_text = text[:30] + "..." if len(text) > 30 else text
        console.print(f"[bold cyan]⌨️  TYPE[/bold cyan] '[bold yellow]{display_text}[/bold yellow]'")

        escaped_text = self.__escape(text=text)
        return await self.__shell(command=f'input text "{escaped_text}"')

    async def swipe(
        self, *, x1: int, y1: int, x2: int, y2: int, duration: int = 300
    ) -> ActionResult:
        """
        Execute swipe gesture.
        """

        console.print(
            f"[bold cyan]↔️  SWIPE[/bold cyan] from ([bold yellow]{x1}, {y1}[/bold yellow]) "
            f"to ([bold yellow]{x2}, {y2}[/bold yellow]) in {duration}ms"
        )

        time_limit = duration or self.__configuration.swipe_duration
        return await self.__shell(command=f"input swipe {x1} {y1} {x2} {y2} {time_limit}")

    async def back(self) -> ActionResult:
        """
        Press back button.
        """

        console.print("[bold cyan]⬅️  BACK[/bold cyan] button")
        return await self.__keyevent(keycode=4)

    async def home(self) -> ActionResult:
        """
        Press home button.
        """

        console.print("[bold cyan]🏠  HOME[/bold cyan] button")
        return await self.__keyevent(keycode=3)

    async def get_screen_size(self) -> Tuple[int, int]:
        """
        Get device screen dimensions.
        """

        if self.__cached_size:
            return self.__cached_size

        result = await self.__shell(command="wm size", capture_output=True)
        if not result.success or not result.output:
            return (1080, 1920)

        if match := re.search(r"(\d+)x(\d+)", result.output):
            width = int(match.group(1))
            height = int(match.group(2))
            self.__cached_size = (width, height)
            return width, height

        return (1080, 1920)

    async def capture_screen(self) -> bytes:
        """
        Capture device screenshot.
        """

        arguments = self.__build_arguments(parts=["exec-out", "screencap", "-p"])

        try:
            process = await asyncio.create_subprocess_exec(
                *arguments,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(
                fut=process.communicate(),
                timeout=self.__configuration.command_timeout,
            )
            return stdout if process.returncode == 0 and stdout else b""
        except Exception:
            return b""

    async def get_current_package(self) -> str:
        """
        Get current foreground package name.
        """

        result = await self.__shell(
            command="dumpsys activity activities | grep mResumedActivity", capture_output=True
        )

        if (
            result.success
            and result.output
            and (match := re.search(r"u0\s+([a-zA-Z0-9_.]+)/", result.output))
        ):
            return match.group(1)

        return "unknown_app"

    async def wait_for_device(self, *, timeout: float = 30.0) -> bool:
        """
        Wait for device availability.
        """

        try:
            process = await asyncio.create_subprocess_exec(
                *self.__build_arguments(parts=["wait-for-device"]),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(fut=process.wait(), timeout=timeout)
            return process.returncode == 0
        except Exception:
            return False

    async def dump_hierarchy(self) -> Optional[str]:
        """
        Dump UI hierarchy to XML string.
        """

        path = "/data/local/tmp/window_dump.xml"
        dump_command = f"uiautomator dump --compressed {path}"
        dump_result = await self.__shell(command=dump_command)

        if not dump_result.success:
            return None

        cat_arguments = self.__build_arguments(parts=["exec-out", "cat", path])
        try:
            process = await asyncio.create_subprocess_exec(
                *cat_arguments,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(fut=process.communicate(), timeout=5.0)
            return stdout.decode("utf-8", errors="ignore") if stdout else None
        except Exception:
            return None

    async def get_snapshot(self) -> Tuple[bytes, Optional[str]]:
        """
        Capture atomic snapshot (Screenshot + XML) in parallel.
        """

        results = await asyncio.gather(
            self.capture_screen(), self.dump_hierarchy(), return_exceptions=True
        )

        image = results[0] if not isinstance(results[0], Exception) else b""
        xml = results[1] if not isinstance(results[1], Exception) else None

        return image, xml

    # Helper methods copied from original tool
    async def __shell(self, command: str, *, capture_output: bool = False) -> ActionResult:
        """
        Execute ADB shell command with terminal logging.
        """

        arguments = self.__build_arguments(parts=["shell", command])
        start_time = asyncio.get_event_loop().time()

        try:
            process = await asyncio.create_subprocess_exec(
                *arguments,
                stderr=asyncio.subprocess.PIPE,
                stdout=(asyncio.subprocess.PIPE if capture_output else asyncio.subprocess.DEVNULL),
            )

            stdout, stderr = await asyncio.wait_for(
                fut=process.communicate(),
                timeout=self.__configuration.command_timeout,
            )

            duration = int((asyncio.get_event_loop().time() - start_time) * 1000)

            # Rich formatting for command logs
            color_theme = "green" if process.returncode == 0 else "red"
            console.print(
                f"[bold blue]⚡ ADB[/bold blue] [white]❯[/white] "
                f"[{color_theme}]{command[:100]}{'...' if len(command) > 100 else ''}[/{color_theme}] "
                f"[bold yellow]{duration}ms[/bold yellow]"
            )

            if process.returncode != 0:
                error_message = stderr.decode().strip() if stderr else "Failed"
                return ActionResult(success=False, error=error_message, duration=duration)

            return ActionResult(
                success=True,
                output=stdout.decode().strip() if stdout else None,
                duration=duration,
            )

        except Exception as exception:
            return ActionResult(success=False, error=str(exception), duration=0)

    async def __keyevent(self, keycode: int) -> ActionResult:
        """
        Execute a key event.
        """

        return await self.__shell(command=f"input keyevent {keycode}")

    def __build_arguments(self, parts: List[str]) -> List[str]:
        """
        Builds full command list.
        """

        cmd = [self.__configuration.adb_path]
        if self.__configuration.device_serial:
            cmd.extend(["-s", self.__configuration.device_serial])

        cmd.extend(parts)
        return cmd

    def __escape(self, text: str) -> str:
        """
        Escapes text for ADB.
        """

        return text.replace(r"\\", r"\\\\").replace(r'"', r"\"").replace(r" ", r"%s")
