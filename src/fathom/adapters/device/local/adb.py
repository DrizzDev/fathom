from __future__ import annotations

import asyncio
import re
from logging import getLogger
from typing import List, Optional, Tuple

from fathom.constants.interaction import SwipeSpeed
from fathom.constants.platform import DevicePlatform
from fathom.core.exceptions import DeviceError
from fathom.interfaces.device import DevicePort
from fathom.schemas.configuration import (
    ADBConfiguration,
    DeviceRuntimeConfiguration,
    InteractionPolicyConfiguration,
    InteractionRuntimeConfiguration,
    ScrollInteractionPolicy,
    SwipeInteractionPolicy,
)
from fathom.schemas.results import ActionResult

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

        self.__runtime_configuration = DeviceRuntimeConfiguration(
            platform=DevicePlatform.ANDROID,
            identifier=self.__configuration.serial_number,
            command_timeout=self.__configuration.command_timeout,
            interaction=InteractionRuntimeConfiguration(
                policy=InteractionPolicyConfiguration(
                    swipe=SwipeInteractionPolicy(
                        duration_milliseconds=(
                            self.__configuration.interaction.policy.swipe.duration_milliseconds
                        ),
                        distance_ratio=self.__configuration.interaction.policy.swipe.distance_ratio,
                    ),
                    scroll=ScrollInteractionPolicy(
                        distance_ratio=self.__configuration.interaction.policy.scroll.distance_ratio
                    ),
                )
            ),
            metadata={"executable_path": self.__configuration.executable_path},
        )
        self.__cached_size: Optional[Tuple[int, int]] = None
        self.__launch_attempted: bool = False

    @property
    def configuration(self) -> DeviceRuntimeConfiguration:
        """
        Returns the tool configuration.
        """

        return self.__runtime_configuration

    async def tap(self, *, x: int, y: int) -> ActionResult:
        """
        Execute tap at coordinates.
        """

        return await self.__shell(command=f"input tap {x} {y}")

    async def type(self, *, text: str) -> ActionResult:
        """
        Type text on device character-by-character to prevent ADB input drops.

        ``__execute_type`` taps the text field to focus it immediately
        before calling this method, but Android needs a brief moment to
        route the focus event to the IME + open the input connection.
        Without a guard the first few characters of the stream can
        arrive before the field is ready and get dropped. 200 ms
        covers slower-to-focus fields (search bars that expand on tap,
        inputs inside modals) without being humanly noticeable.
        """

        await asyncio.sleep(0.2)

        last_result = ActionResult(success=True, duration=0)

        for character in text:
            escaped = self.__escape(text=character)

            if (result := await self.__shell(command=f'input text "{escaped}"')).success is False:
                return result

            last_result = result
            await asyncio.sleep(0.01)

        return last_result

    async def swipe(
        self,
        *,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        duration: Optional[int] = None,
        speed: Optional[SwipeSpeed] = None,
    ) -> ActionResult:
        """
        Execute swipe gesture.
        """

        if speed is not None:
            logger.debug("Ignoring swipe speed for ADB adapter: %s", speed)

        duration = duration or (
            self.__configuration.interaction.policy.swipe.duration_milliseconds
            if self.__configuration
            else 300
        )

        return await self.__shell(command=f"input swipe {x1} {y1} {x2} {y2} {duration}")

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

    async def __run_safe_subprocess(
        self,
        arguments: List[str],
        *,
        timeout: float,
        capture_stdout: bool = True,
        capture_stderr: bool = True,
    ) -> Tuple[int, bytes, bytes]:
        """
        Centralized, safe subprocess executor.
        Guarantees that underlying OS processes are killed if the operation times out
        or if the Python task is cancelled, preventing ADB connection deadlocks.
        """

        process = None

        try:
            process = await asyncio.create_subprocess_exec(
                *arguments,
                stdout=asyncio.subprocess.PIPE if capture_stdout else asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE if capture_stderr else asyncio.subprocess.DEVNULL,
            )

            stdout, stderr = await asyncio.wait_for(
                fut=process.communicate(),
                timeout=timeout,
            )

            return process.returncode or 0, stdout, stderr

        except asyncio.TimeoutError as exception:
            raise DeviceError(
                f"Command timed out after {timeout}s: {' '.join(arguments)}"
            ) from exception

        except Exception as exception:
            raise DeviceError(f"Command execution failed: {exception}") from exception

        finally:
            if process and process.returncode is None:
                try:
                    process.kill()
                    await process.wait()
                except ProcessLookupError:
                    pass  # Process already dead
                except Exception as cleanup_exception:
                    logger.warning(f"Failed to cleanup subprocess: {cleanup_exception}")

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

        # Invalidate cached dimensions so that any subsequent
        # get_dimensions() call re-queries the OS.  This prevents
        # stale portrait dimensions from persisting after a rotation.
        self.__cached_size = None

        arguments = self.__build_arguments(parts=["exec-out", "screencap", "-p"])

        try:
            returncode, stdout, stderr = await self.__run_safe_subprocess(
                arguments=arguments,
                timeout=self.__configuration.command_timeout,
                capture_stdout=True,
                capture_stderr=True,
            )

            if returncode != 0:
                error_msg = stderr.decode().strip() if stderr else "Unknown error"
                raise DeviceError(f"Screenshot capture failed: {error_msg}")

            if not stdout:
                raise DeviceError("Screenshot capture returned empty data")

            return stdout

        except DeviceError:
            raise
        except Exception as exception:
            raise DeviceError(f"Screenshot capture failed: {exception}") from exception

    async def launch_configured_package(self) -> None:
        """
        Launch the configured Android package exactly once per session.

        Called from ``IntentStrategy.execute`` as a background task so
        the launch runs concurrently with LLM-based intent
        classification and decomposition — the app is typically ready
        by the time the agent takes its first screenshot.

        Uses ``monkey -p <pkg> -c android.intent.category.LAUNCHER 1``
        which resolves the launcher activity itself, so the adapter
        doesn't need to know the app's entry Activity. Failures are
        logged and swallowed — the agent can still navigate from the
        launcher if the auto-launch fails (e.g., package not installed).
        """

        if self.__launch_attempted:
            return

        package_name = self.__configuration.package_name
        if not package_name:
            return

        self.__launch_attempted = True

        arguments = self.__build_arguments(
            parts=[
                "shell",
                "monkey",
                "-p",
                package_name,
                "-c",
                "android.intent.category.LAUNCHER",
                "1",
            ],
        )

        try:
            returncode, _stdout, stderr = await self.__run_safe_subprocess(
                arguments=arguments,
                timeout=self.__configuration.command_timeout,
                capture_stdout=True,
                capture_stderr=True,
            )
            if returncode != 0:
                stderr_text = stderr.decode("utf-8", errors="ignore").strip() if stderr else ""
                logger.warning(
                    "[adb] auto-launch of %s failed (exit=%s): %s; agent will navigate from launcher",
                    package_name,
                    returncode,
                    stderr_text or "no stderr",
                )
                return
            logger.info("[adb] auto-launched %s", package_name)
        except Exception as exception:
            logger.warning(
                "[adb] auto-launch of %s raised %s; agent will navigate from launcher",
                package_name,
                exception,
            )

    async def terminate_configured_package(self) -> None:
        """
        Stop the configured Android package on run exit.

        Uses ``adb shell am force-stop <pkg>``. Safe on apps that are
        not running (force-stop is a no-op in that case). Failures are
        logged and swallowed — cleanup must never raise.
        """

        package_name = self.__configuration.package_name
        if not package_name:
            return

        arguments = self.__build_arguments(
            parts=["shell", "am", "force-stop", package_name],
        )

        try:
            returncode, _stdout, stderr = await self.__run_safe_subprocess(
                arguments=arguments,
                timeout=self.__configuration.command_timeout,
                capture_stdout=True,
                capture_stderr=True,
            )
            if returncode != 0:
                stderr_text = stderr.decode("utf-8", errors="ignore").strip() if stderr else ""
                logger.warning(
                    "[adb] terminate of %s failed (exit=%s): %s",
                    package_name,
                    returncode,
                    stderr_text or "no stderr",
                )
                return
            logger.info("[adb] terminated %s", package_name)
        except Exception as exception:
            logger.warning(
                "[adb] terminate of %s raised %s",
                package_name,
                exception,
            )

    async def get_current_package(self) -> str:
        """
        Get current foreground package name.
        """

        result = await self.__shell(
            command="dumpsys window | grep mCurrentFocus", capture_output=True
        )

        if not result.success:
            raise DeviceError(
                f"Get current package: Failed to get current package: {result.error or 'Unknown error'}"
            )

        if not result.output:
            raise DeviceError("Get current package: Package query returned empty output")

        # Expected format: mCurrentFocus=Window{e5fb16 u0 com.package.name/com.package.name.MainActivity}
        if match := re.search(
            r"mCurrentFocus=Window\{[a-f0-9]+\s+(?:u\d+\s+)?([a-zA-Z0-9_.]+)\/", result.output
        ):
            return match.group(1)

        raise DeviceError(
            f"Get current package: Failed to parse package name from output: {result.output}"
        )

    async def wait_for_device(self, *, timeout: float = 30.0) -> bool:
        """
        Wait for device availability.
        """

        arguments = self.__build_arguments(parts=["wait-for-device"])

        try:
            returncode, stdout_bytes, stderr_bytes = await self.__run_safe_subprocess(
                arguments=arguments,
                timeout=timeout,
                capture_stdout=False,
                capture_stderr=False,
            )
            if stdout_bytes or stderr_bytes:
                logger.debug("wait-for-device produced subprocess output unexpectedly")
            return returncode == 0
        except Exception:
            return False

    async def dump_hierarchy(self) -> Optional[str]:
        """
        Dump UI hierarchy to XML string.
        Attempts compressed dump first, with fallback to uncompressed and process cleanup.
        """

        path = "/data/local/tmp/window_dump.xml"

        # Ensure we don't read a stale file
        await self.__shell(command=f"rm -f {path}")

        dump_command = f"uiautomator dump --compressed {path}"
        dump_result = await self.__shell(command=dump_command)

        if not dump_result.success:
            logger.warning(
                f"Compressed dump failed: {dump_result.error}. Attempting recovery and fallback."
            )
            # Device-side recovery: forcefully kill hung uiautomator service
            await self.__shell(command="pkill -9 uiautomator")

            # Fallback to uncompressed dump
            fallback_command = f"uiautomator dump {path}"
            dump_result = await self.__shell(command=fallback_command)

            if not dump_result.success:
                raise DeviceError(
                    f"Dump hierarchy: UI automation dump failed on device after fallback: {dump_result.error or 'Unknown error'}"
                )

        cat_arguments = self.__build_arguments(parts=["exec-out", "cat", path])

        try:
            returncode, stdout, stderr = await self.__run_safe_subprocess(
                arguments=cat_arguments,
                timeout=10.0,
                capture_stdout=True,
                capture_stderr=True,
            )

            if returncode != 0:
                error_msg = stderr.decode().strip() if stderr else "Unknown error"
                raise DeviceError(f"Dump hierarchy: Failed to read hierarchy file: {error_msg}")

            if not stdout:
                raise DeviceError("Dump hierarchy: Hierarchy dump returned empty data")

            return stdout.decode("utf-8", errors="ignore")

        except DeviceError:
            raise
        except Exception as exception:
            raise DeviceError(
                f"Dump hierarchy: Unexpected error during XML retrieval: {exception}"
            ) from exception

    async def get_snapshot(self) -> Tuple[bytes, Optional[str]]:
        """
        Capture atomic snapshot (Screenshot + XML) in parallel.
        """

        results = await asyncio.gather(
            self.capture_screen(),
            self.dump_hierarchy(),
            return_exceptions=True,
        )

        image_result = results[0]
        xml_result = results[1]

        if isinstance(xml_result, Exception):
            logger.error(f"Snapshot: Hierarchy dump failed: {xml_result}")

        if isinstance(image_result, Exception):
            logger.error(f"Snapshot: Screenshot capture failed: {image_result}")
            raise DeviceError(f"Snapshot capture failed: {image_result}") from image_result

        image = image_result if isinstance(image_result, bytes) else b""
        xml = xml_result if isinstance(xml_result, str) else None

        return image, xml

    # Helper methods copied from original tool
    async def __shell(self, command: str, *, capture_output: bool = False) -> ActionResult:
        """
        Execute ADB shell command with terminal logging.
        """

        arguments = self.__build_arguments(parts=["shell", command])
        start_time = asyncio.get_event_loop().time()

        try:
            returncode, stdout, stderr = await self.__run_safe_subprocess(
                arguments=arguments,
                timeout=self.__configuration.command_timeout,
                capture_stdout=capture_output,
                capture_stderr=True,
            )

            duration = int((asyncio.get_event_loop().time() - start_time) * 1000)

            if returncode != 0:
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

        return text.replace("\\", "\\\\").replace('"', '\\"').replace(" ", "%s")
