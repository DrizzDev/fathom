from __future__ import annotations

import asyncio
import re
from logging import getLogger
from typing import List, Optional, Tuple

from rich.console import Console

from fathom.schemas.configuration import ADBConfig
from fathom.schemas.results import ActionResult
from fathom.tools.device.base import DeviceTool

logger = getLogger(__name__)
console = Console()


class ADBDeviceTool(DeviceTool):
    """
    Device tool using Android Debug Bridge (ADB).
    """

    def __init__(self, configuration: Optional[ADBConfig] = None) -> None:
        self.__configuration = configuration or ADBConfig()
        self.__cached_size: Optional[Tuple[int, int]] = None

    async def tap(self, x: int, y: int) -> ActionResult:
        """
        Execute tap at coordinates.
        """
        return await self.__shell(f"input tap {x} {y}")

    async def type_text(self, text: str) -> ActionResult:
        """
        Type text on device.
        """
        escaped = self.__escape(text)
        return await self.__shell(f'input text "{escaped}"')

    async def swipe(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        duration: Optional[int] = None,
    ) -> ActionResult:
        """
        Execute swipe gesture.
        """
        time_limit = duration or self.__configuration.swipe_duration
        return await self.__shell(f"input swipe {x1} {y1} {x2} {y2} {time_limit}")

    async def back(self) -> ActionResult:
        """
        Press back button.
        """
        return await self.__keyevent(4)

    async def home(self) -> ActionResult:
        """
        Press home button.
        """
        return await self.__keyevent(3)

    async def get_screen_size(self) -> Tuple[int, int]:
        """
        Get device screen dimensions.
        """
        if self.__cached_size:
            return self.__cached_size

        result = await self.__shell("wm size", capture_output=True)
        if not result.success or not result.output:
            return (1080, 1920)

        match = re.search(r"(\d+)x(\d+)", result.output)
        if match:
            width = int(match.group(1))
            height = int(match.group(2))
            self.__cached_size = (width, height)
            return width, height

        return (1080, 1920)

    async def screenshot(self) -> Optional[bytes]:
        """
        Capture device screenshot.
        """
        arguments = self.__build_arguments(["exec-out", "screencap", "-p"])
        try:
            process = await asyncio.create_subprocess_exec(
                *arguments,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(
                process.communicate(),
                timeout=self.__configuration.command_timeout,
            )
            return stdout if process.returncode == 0 and stdout else None
        except Exception:
            return None

    async def dump_hierarchy(self) -> Optional[str]:
        """
        Dump UI hierarchy to XML string efficiently.
        """
        # Try direct stdout dump first (fastest, no temp files)
        # Note: 'uiautomator dump /dev/stdout' works on many modern Android versions
        arguments = self.__build_arguments(
            ["shell", "uiautomator", "dump", "--compressed", "/dev/stdout"]
        )
        try:
            process = await asyncio.create_subprocess_exec(
                *arguments,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=5.0,
            )
            if process.returncode == 0 and stdout:
                output = stdout.decode("utf-8", errors="ignore")
                if "UI hierchary dumped to:" in output:
                    # If it didn't actually dump to stdout but to a file anyway
                    pass
                else:
                    # Clean up any potential 'UI hierchary dumped to' prefix if it exists
                    match = re.search(r"(<hierarchy.*</hierarchy>)", output, re.DOTALL)
                    if match:
                        return match.group(1)
        except Exception as exception:
            logger.debug(f"Direct XML dump failed, falling back: {exception}")

        # Fallback to temp file if stdout dump fails
        path = "/data/local/tmp/window_dump.xml"
        dump = await self.__shell(f"uiautomator dump --compressed {path}")
        if not dump.success:
            return None

        # Use exec-out cat for slightly better performance than shell cat
        cat_args = self.__build_arguments(["exec-out", "cat", path])
        try:
            process = await asyncio.create_subprocess_exec(
                *cat_args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(
                process.communicate(),
                timeout=5.0,
            )
            return stdout.decode("utf-8", errors="ignore") if stdout else None
        except Exception:
            return None

    async def execute(self, request: dict[str, object]) -> ActionResult:
        """
        Execute generic action from request dictionary.
        """
        name = str(request.get("action", ""))

        if name == "tap":
            return await self.tap(int(str(request.get("x", 0))), int(str(request.get("y", 0))))

        if name == "type":
            return await self.type_text(str(request.get("text", "")))

        if name == "swipe":
            x1 = int(str(request.get("x1", 0)))
            y1 = int(str(request.get("y1", 0)))
            x2 = int(str(request.get("x2", 0)))
            y2 = int(str(request.get("y2", 0)))
            duration = request.get("duration")
            return await self.swipe(x1, y1, x2, y2, int(str(duration)) if duration else None)

        if name == "back":
            return await self.back()

        if name == "home":
            return await self.home()

        if name == "wait":
            duration = int(str(request.get("duration", 1000)))
            await asyncio.sleep(duration / 1000)
            return ActionResult(success=True, duration=duration)

        return (
            ActionResult(success=True, duration=0)
            if name == "complete"
            else ActionResult(success=False, error=f"Unknown: {name}", duration=0)
        )

    async def __shell(self, command: str, *, capture_output: bool = False) -> ActionResult:
        """
        Execute ADB shell command with terminal logging.
        """
        arguments = self.__build_arguments(["shell", command])
        start = asyncio.get_event_loop().time()

        try:
            process = await asyncio.create_subprocess_exec(
                *arguments,
                stderr=asyncio.subprocess.PIPE,
                stdout=(asyncio.subprocess.PIPE if capture_output else asyncio.subprocess.DEVNULL),
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.__configuration.command_timeout,
            )

            duration = int((asyncio.get_event_loop().time() - start) * 1000)

            # Rich formatting for command logs
            color = "green" if process.returncode == 0 else "red"
            console.print(
                f"[bold blue]⚡ ADB[/bold blue] [white]❯[/white] "
                f"[{color}]{command[:60]}{'...' if len(command) > 60 else ''}[/{color}] "
                f"[bold yellow]{duration}ms[/bold yellow]"
            )

            if process.returncode != 0:
                error = stderr.decode().strip() if stderr else "Failed"
                return ActionResult(success=False, error=error, duration=duration)

            return ActionResult(
                success=True, output=stdout.decode().strip() if stdout else None, duration=duration
            )

        except Exception as exception:
            return ActionResult(success=False, error=str(exception), duration=0)

    async def __keyevent(self, keycode: int) -> ActionResult:
        """
        Execute a key event.
        """
        return await self.__shell(f"input keyevent {keycode}")

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

    async def wait_for_device(self, timeout: float = 30.0) -> bool:
        """
        Wait for device availability.
        """
        try:
            process = await asyncio.create_subprocess_exec(
                *self.__build_arguments(["wait-for-device"]),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(process.wait(), timeout=timeout)
            return process.returncode == 0
        except Exception:
            return False

    async def cleanup(self) -> None:
        """
        Cleanup logic.
        """
        pass
