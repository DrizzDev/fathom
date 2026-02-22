from __future__ import annotations

import asyncio
import re
from logging import getLogger
from typing import List, Optional, Tuple

from rich.console import Console

from fathom.core.exceptions import DeviceError
from fathom.interfaces.device import DevicePort
from fathom.schemas.configuration import ADBConfiguration
from fathom.schemas.results import ActionResult

console = Console()
logger = getLogger(__name__)


class ADBDevice(DevicePort):
    """
    ADB adapter for Local Android devices.
    """

    def __init__(
        self,
        *,
        serial: Optional[str] = None,
        configuration: Optional[ADBConfiguration] = None,
    ) -> None:
        """
        Initialize ADB device adapter.
        """

        if configuration:
            self.__configuration = configuration

        else:
            self.__configuration = (
                ADBConfiguration(serial_number=serial) if serial else ADBConfiguration()
            )

        self.__cached_size: Optional[Tuple[int, int]] = None

    @property
    def configuration(self) -> ADBConfiguration:
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

    async def type(self, *, text: str) -> ActionResult:
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

    async def get_dimensions(self) -> Tuple[int, int]:
        """
        Get device screen dimensions.
        """

        if self.__cached_size:
            return self.__cached_size

        result = await self.__shell(command="wm size", capture_output=True)

        if not result.success:
            raise DeviceError(
                f"Get dimensions: Failed to get screen dimensions: {result.error or 'Unknown error'}"
            )

        if not result.output:
            raise DeviceError("Get dimensions: Screen dimension command returned empty output")

        if match := re.search(r"(\d+)x(\d+)", result.output):
            width = int(match.group(1))
            height = int(match.group(2))
            self.__cached_size = (width, height)
            return width, height

        raise DeviceError(
            f"Get dimensions: Failed to parse screen dimensions from output: {result.output}"
        )

    async def capture_screen(self) -> bytes:
        """
        Capture device screenshot.
        """

        arguments = self.__build_arguments(parts=["exec-out", "screencap", "-p"])

        process = None
        try:
            process = await asyncio.create_subprocess_exec(
                *arguments,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                fut=process.communicate(),
                timeout=self.__configuration.command_timeout,
            )

            if process.returncode != 0:
                error_msg = stderr.decode().strip() if stderr else "Unknown error"
                raise DeviceError(f"Screenshot capture failed: {error_msg}")

            if not stdout:
                raise DeviceError("Screenshot capture returned empty data")

            return stdout

        except asyncio.TimeoutError as exception:
            raise DeviceError(
                f"Screenshot capture timed out after {self.__configuration.command_timeout}s"
            ) from exception

        except DeviceError:
            raise

        except Exception as exception:
            raise DeviceError(f"Screenshot capture failed: {exception}") from exception

        finally:
            # Ensure process is terminated if still running
            if process and process.returncode is None:
                try:
                    process.kill()
                    await process.wait()
                except Exception as cleanup_exception:
                    logger.warning(f"Failed to cleanup subprocess: {cleanup_exception}")

    async def get_current_package(self) -> str:
        """
        Get current foreground package name.
        """

        result = await self.__shell(
            command="dumpsys activity activities | grep mResumedActivity", capture_output=True
        )

        if not result.success:
            raise DeviceError(
                f"Get current package: Failed to get current package: {result.error or 'Unknown error'}"
            )

        if not result.output:
            raise DeviceError("Get current package: Package query returned empty output")

        if match := re.search(r"u0\s+([a-zA-Z0-9_.]+)/", result.output):
            return match.group(1)

        raise DeviceError(
            f"Get current package: Failed to parse package name from output: {result.output}"
        )

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
            raise DeviceError(
                f"Dump hierarchy: Failed to dump UI hierarchy: {dump_result.error or 'Unknown error'}"
            )

        cat_arguments = self.__build_arguments(parts=["exec-out", "cat", path])

        try:
            process = await asyncio.create_subprocess_exec(
                *cat_arguments,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(fut=process.communicate(), timeout=5.0)

            if process.returncode != 0:
                error_msg = stderr.decode().strip() if stderr else "Unknown error"
                raise DeviceError(f"Dump hierarchy: Failed to read hierarchy file: {error_msg}")

            if not stdout:
                raise DeviceError("Dump hierarchy: Hierarchy dump returned empty data")

            return stdout.decode("utf-8", errors="ignore")

        except asyncio.TimeoutError as exception:
            raise DeviceError("Dump hierarchy: Hierarchy dump timed out after 5s") from exception

        except DeviceError:
            raise

        except Exception as exception:
            raise DeviceError(
                f"Dump hierarchy: Failed to dump hierarchy: {exception}"
            ) from exception

    async def get_snapshot(self) -> Tuple[bytes, Optional[str], Tuple[int, int]]:
        """
        Capture atomic snapshot (Screenshot + XML + Dimensions) in parallel.
        """

        results = await asyncio.gather(
            self.capture_screen(),
            self.dump_hierarchy(),
            self.get_dimensions(),
            return_exceptions=True,
        )

        image_result = results[0]
        xml_result = results[1]
        dim_result = results[2]

        image = image_result if isinstance(image_result, bytes) else b""
        xml = xml_result if isinstance(xml_result, str) else None
        dimensions = dim_result if isinstance(dim_result, tuple) else (1080, 1920)

        return image, xml, dimensions

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
                duration=duration,
                output=stdout.decode().strip() if stdout else None,
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

        cmd = [self.__configuration.executable_path]
        if self.__configuration.serial_number:
            cmd.extend(["-s", self.__configuration.serial_number])

        cmd.extend(parts)
        return cmd

    def __escape(self, text: str) -> str:
        """
        Escapes text for ADB.
        """

        return text.replace(r"\\", r"\\\\").replace(r'"', r"\"").replace(r" ", r"%s")
