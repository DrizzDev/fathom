from __future__ import annotations

import asyncio
import re
from typing import List, Optional, Tuple

from fathom.schemas.configuration import ADBConfig
from fathom.schemas.results import ActionResult
from fathom.tools.device.base import DeviceTool


class ADBDeviceTool(DeviceTool):
    """Device tool using Android Debug Bridge (ADB).

    Implements real device control via ADB shell commands.
    Supports physical devices and emulators.

    Example:
        ```python
        adb = ADBDeviceTool(ADBConfig(device_serial="emulator-5554"))
        result = await adb.tap(500, 500)
        if result.success:
            print("Tap executed")
        ```
    """

    def __init__(self, config: Optional[ADBConfig] = None) -> None:
        """Initialize ADB device tool.

        Args:
            config: ADB configuration. Defaults to auto-detect device.
        """
        self.__config = config or ADBConfig()
        self.__cached_screen_size: Optional[Tuple[int, int]] = None

    async def tap(self, x: int, y: int) -> ActionResult:
        """Execute tap at coordinates.

        Args:
            x: X coordinate in pixels.
            y: Y coordinate in pixels.

        Returns:
            ActionResult indicating success or failure.
        """
        cmd = f"input tap {x} {y}"
        return await self.__shell(cmd)

    async def type(self, text: str) -> ActionResult:
        """Type text on device.

        Args:
            text: Text to type.

        Returns:
            ActionResult.
        """
        escaped = self.__escape_text(text)
        cmd = f'input text "{escaped}"'
        return await self.__shell(cmd)

    async def swipe(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        duration: Optional[int] = None,
    ) -> ActionResult:
        """Execute swipe gesture.

        Args:
            x1, y1: Start coordinates.
            x2, y2: End coordinates.
            duration: Swipe duration in milliseconds.

        Returns:
            ActionResult.
        """
        duration_ms = duration or self.__config.swipe_duration
        cmd = f"input swipe {x1} {y1} {x2} {y2} {duration_ms}"
        return await self.__shell(cmd)

    async def back(self) -> ActionResult:
        """Press back button.

        Returns:
            ActionResult.
        """
        return await self.__keyevent(4)

    async def home(self) -> ActionResult:
        """Press home button.

        Returns:
            ActionResult.
        """
        return await self.__keyevent(3)

    async def get_screen_size(self) -> Tuple[int, int]:
        """Get device screen dimensions.

        Returns:
            Tuple of (width, height) in pixels.
        """
        if self.__cached_screen_size:
            return self.__cached_screen_size

        result = await self.__shell("wm size", capture_output=True)

        if not result.success or not result.output:
            return (1080, 1920)

        match = re.search(r"(\d+)x(\d+)", result.output)
        if match:
            width = int(match.group(1))
            height = int(match.group(2))
            self.__cached_screen_size = (width, height)
            return (width, height)

        return (1080, 1920)

    async def get_current_activity(self) -> str:
        """Get current foreground activity.

        Returns:
            Activity name in format package/activity.
        """
        cmd = "dumpsys activity activities | grep mResumedActivity"
        result = await self.__shell(cmd, capture_output=True)

        if not result.success or not result.output:
            return "unknown"

        match = re.search(r"(\S+/\S+)", result.output)
        if match:
            return match.group(1)

        return "unknown"

    async def execute(self, request: dict[str, object]) -> ActionResult:
        """Execute action from request dictionary.

        Args:
            request: Action request with 'action' key and parameters.

        Returns:
            ActionResult.
        """
        action = str(request.get("action", ""))

        if action == "tap":
            x = int(request.get("x", 0))  # type: ignore
            y = int(request.get("y", 0))  # type: ignore
            return await self.tap(x, y)

        if action == "type":
            text = str(request.get("text", ""))
            return await self.type(text)

        if action == "swipe":
            x1 = int(request.get("x1", 0))  # type: ignore
            y1 = int(request.get("y1", 0))  # type: ignore
            x2 = int(request.get("x2", 0))  # type: ignore
            y2 = int(request.get("y2", 0))  # type: ignore
            duration_raw = request.get("duration")
            duration = int(duration_raw) if duration_raw else None  # type: ignore
            return await self.swipe(x1, y1, x2, y2, duration)

        if action == "back":
            return await self.back()

        if action == "home":
            return await self.home()

        if action == "wait":
            duration_raw = request.get("duration", 1000)
            duration_ms = int(duration_raw) if duration_raw else 1000  # type: ignore
            await asyncio.sleep(duration_ms / 1000)
            return ActionResult(success=True, duration=duration_ms)

        if action == "complete":
            return ActionResult(success=True, duration=0)

        return ActionResult(
            success=False,
            error=f"Unknown action: {action}",
            duration=0,
        )

    async def __shell(
        self,
        command: str,
        *,
        capture_output: bool = False,
    ) -> ActionResult:
        """Execute ADB shell command.

        Args:
            command: Shell command to execute.
            capture_output: Whether to capture and return output.

        Returns:
            ActionResult with optional output.
        """
        args = self.__build_adb_args(["shell", command])

        try:
            process = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE if capture_output else asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.__config.command_timeout,
            )

            if process.returncode != 0:
                error_msg = stderr.decode().strip() if stderr else "Command failed"
                return ActionResult(success=False, error=error_msg, duration=0)

            output = stdout.decode().strip() if stdout else None
            return ActionResult(success=True, output=output, duration=0)

        except asyncio.TimeoutError:
            return ActionResult(
                success=False,
                error=f"Command timed out after {self.__config.command_timeout}s",
                duration=int(self.__config.command_timeout * 1000),
            )
        except FileNotFoundError:
            return ActionResult(
                success=False,
                error=f"ADB not found at: {self.__config.adb_path}",
                duration=0,
            )
        except Exception as exception:
            return ActionResult(success=False, error=str(exception), duration=0)

    async def __keyevent(self, keycode: int) -> ActionResult:
        """Send key event.

        Args:
            keycode: Android keycode.

        Returns:
            ActionResult.
        """
        return await self.__shell(f"input keyevent {keycode}")

    def __build_adb_args(self, args: List[str]) -> List[str]:
        """Build full ADB command arguments.

        Args:
            args: ADB subcommand and arguments.

        Returns:
            Full command list with adb path and device serial.
        """
        cmd = [self.__config.adb_path]
        if self.__config.device_serial:
            cmd.extend(["-s", self.__config.device_serial])
        cmd.extend(args)
        return cmd

    def __escape_text(self, text: str) -> str:
        """Escape text for ADB input.

        Args:
            text: Raw text.

        Returns:
            Escaped text safe for shell.
        """
        return text.replace("\\", "\\\\").replace('"', '\\"').replace(" ", "%s")

    async def is_connected(self) -> bool:
        """Check if device is connected.

        Returns:
            True if device responds to ADB.
        """
        args = self.__build_adb_args(["get-state"])

        try:
            process = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(
                process.communicate(),
                timeout=5.0,
            )
            return b"device" in stdout

        except Exception:
            return False

    async def wait_for_device(self, timeout: float = 30.0) -> bool:
        """Wait for device to become available.

        Args:
            timeout: Maximum wait time in seconds.

        Returns:
            True if device became available.
        """
        args = self.__build_adb_args(["wait-for-device"])

        try:
            process = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(process.wait(), timeout=timeout)
            return process.returncode == 0

        except asyncio.TimeoutError:
            return False

    async def screenshot(self) -> Optional[bytes]:
        """Capture device screenshot.

        Returns:
            PNG image bytes or None on failure.
        """
        args = self.__build_adb_args(["exec-out", "screencap", "-p"])

        try:
            process = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(
                process.communicate(),
                timeout=self.__config.command_timeout,
            )

            if process.returncode == 0 and stdout:
                return stdout

            return None

        except Exception:
            return None
