from __future__ import annotations

import asyncio
import re
from logging import getLogger
from typing import List, Optional, Tuple

from rich.console import Console

from fathom.schemas.configuration import ADBConfig
from fathom.schemas.results import ActionResult
from fathom.tools.device.base import DeviceTool

console = Console()
logger = getLogger(__name__)


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

        return await self.__shell(command=f"input tap {x} {y}")

    async def type_text(self, text: str) -> ActionResult:
        """
        Type text on device.
        """

        escaped_text = self.__escape(text=text)
        return await self.__shell(command=f'input text "{escaped_text}"')

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
        return await self.__shell(command=f"input swipe {x1} {y1} {x2} {y2} {time_limit}")

    async def back(self) -> ActionResult:
        """
        Press back button.
        """

        return await self.__keyevent(keycode=4)

    async def home(self) -> ActionResult:
        """
        Press home button.
        """

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

    async def get_current_package(self) -> str:
        """
        Get current foreground package name.

        Tries multiple strategies to handle different Android versions
        and OEM skins:

        1. ``mResumedActivity`` from ``dumpsys activity`` (most common).
        2. ``mCurrentFocus`` from ``dumpsys window`` (reliable fallback).
        3. ``mFocusedApp`` from ``dumpsys window`` (older devices).
        """

        # Strategy 1: mResumedActivity (Android 10+)
        result = await self.__shell(
            command="dumpsys activity activities | grep mResumedActivity",
            capture_output=True,
        )
        if pkg := self.__extract_package(output=result.output if result.success else None):
            return pkg

        # Strategy 2: mCurrentFocus (window manager — works across most versions)
        result = await self.__shell(
            command="dumpsys window displays | grep mCurrentFocus",
            capture_output=True,
        )
        if pkg := self.__extract_package(output=result.output if result.success else None):
            return pkg

        # Strategy 3: mFocusedApp (older devices)
        result = await self.__shell(
            command="dumpsys window windows | grep mFocusedApp",
            capture_output=True,
        )
        if pkg := self.__extract_package(output=result.output if result.success else None):
            return pkg

        return "unknown_app"

    @staticmethod
    def __extract_package(output: Optional[str]) -> Optional[str]:
        """
        Extract an Android package name from a dumpsys output line.

        Matches the ``com.example.app/...Activity`` pattern found in
        ``mResumedActivity``, ``mCurrentFocus``, and ``mFocusedApp`` lines.
        Requires at least one dot (all real packages have one).
        """

        if not output:
            return None

        # Match "com.example.app/" — package names always contain at least one dot
        match = re.search(r"([a-zA-Z][a-zA-Z0-9_]*(?:\.[a-zA-Z0-9_]+)+)/", output)
        return match.group(1) if match else None

    async def launch_app(self, package_name: str) -> ActionResult:
        """
        Launch an application by package name using monkey.
        """

        return await self.__shell(
            command=f"monkey -p {package_name} -c android.intent.category.LAUNCHER 1"
        )

    async def screenshot(self) -> Optional[bytes]:
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
            return stdout if process.returncode == 0 and stdout else None
        except Exception:
            return None

    async def dump_hierarchy(self) -> Optional[str]:
        """
        Dump UI hierarchy to XML string efficiently.
        """
        path = "/data/local/tmp/window_dump.xml"

        # 1. Dump to file on device (reliable)
        # using nohup or ignoring output can sometimes be faster, but we need to wait for finish
        dump_command = f"uiautomator dump --compressed {path}"
        dump_result = await self.__shell(command=dump_command)

        if not dump_result.success:
            logger.warning("uiautomator dump failed")
            return None

        # 2. Stream file content back (fast)
        cat_arguments = self.__build_arguments(parts=["exec-out", "cat", path])
        try:
            process = await asyncio.create_subprocess_exec(
                *cat_arguments,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(fut=process.communicate(), timeout=5.0)
            return stdout.decode("utf-8", errors="ignore") if stdout else None
        except Exception as exception:
            logger.error(f"Failed to retrieve hierarchy XML: {exception}")
            return None

    async def execute(self, request: dict[str, object]) -> ActionResult:
        """
        Execute generic action from request dictionary.
        """

        name = str(request.get("action", "") or request.get("action_type", ""))

        if name == "tap":
            return await self.tap(x=int(str(request.get("x", 0))), y=int(str(request.get("y", 0))))

        if name == "type":
            return await self.type_text(text=str(request.get("text", "")))

        if name == "swipe":
            x1 = int(str(request.get("x1", 0)))
            y1 = int(str(request.get("y1", 0)))
            x2 = int(str(request.get("x2", 0)))
            y2 = int(str(request.get("y2", 0)))
            duration = request.get("duration")
            return await self.swipe(
                x1=x1, y1=y1, x2=x2, y2=y2, duration=int(str(duration)) if duration else None
            )

        if name == "back":
            return await self.back()

        if name == "home":
            return await self.home()

        if name == "wait":
            duration = int(str(request.get("wait_duration") or request.get("duration") or 1000))
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
                f"[{color_theme}]{command[:60]}{'...' if len(command) > 60 else ''}[/{color_theme}] "
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

    async def wait_for_device(self, timeout: float = 30.0) -> bool:
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

    async def cleanup(self) -> None:
        """
        Cleanup logic.
        """

        pass
