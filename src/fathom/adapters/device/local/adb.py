from __future__ import annotations

import asyncio
import re
from logging import getLogger
from typing import List, Optional, Tuple

from fathom.constants.interaction import SwipeSpeed
from fathom.constants.observation import KeyboardVisibility
from fathom.constants.platform import (
    ANDROID_UIAUTOMATION_ACTIVE_MARKER,
    ANDROID_UIAUTOMATION_DUMP_PATH,
    ANDROID_UIAUTOMATION_INSTRUMENTATION_MARKER,
    ANDROID_UIAUTOMATION_PROCESS_NAME,
    ANDROID_UIAUTOMATION_TIMEOUT_MARKER,
    ANDROID_UIAUTOMATION_UIAUTOMATOR_MARKER,
    AndroidClearStrategy,
    AndroidKeycode,
    DevicePlatform,
)
from fathom.core.exceptions import DeviceError
from fathom.interfaces.device import DevicePort
from fathom.schemas.actions import Bounds, CoordinateSystem
from fathom.schemas.configuration import (
    ADBConfiguration,
    DeviceRuntimeConfiguration,
    InteractionPolicyConfiguration,
    InteractionRuntimeConfiguration,
    ScrollInteractionPolicy,
    SwipeInteractionPolicy,
)
from fathom.schemas.observation import KeyboardObservation
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
                        duration=self.__configuration.interaction.policy.swipe.duration,
                        edge_margin_ratio=(
                            self.__configuration.interaction.policy.swipe.edge_margin_ratio
                        ),
                        minimum_edge_margin=(
                            self.__configuration.interaction.policy.swipe.minimum_edge_margin
                        ),
                        maximum_edge_margin=(
                            self.__configuration.interaction.policy.swipe.maximum_edge_margin
                        ),
                    ),
                    scroll=ScrollInteractionPolicy(
                        edge_margin_ratio=(
                            self.__configuration.interaction.policy.scroll.edge_margin_ratio
                        ),
                        minimum_edge_margin=(
                            self.__configuration.interaction.policy.scroll.minimum_edge_margin
                        ),
                        maximum_edge_margin=(
                            self.__configuration.interaction.policy.scroll.maximum_edge_margin
                        ),
                    ),
                )
            ),
            metadata={"executable_path": self.__configuration.executable_path},
        )
        self.__cached_size: Optional[Tuple[int, int]] = None
        self.__hierarchy_lock = asyncio.Lock()

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

        return await self.__shell(command=f"input touchscreen tap {x} {y}")

    async def type(
        self,
        *,
        text: str,
        prefilled: str = "",
        replace: bool = True,
        locator: Optional[str] = None,
    ) -> ActionResult:
        """
        Type text on device character-by-character to prevent ADB input drops.
        Clears existing content first when *replace* is True and *prefilled* is non-empty.
        """

        _ = locator

        if (
            replace
            and len(prefilled) > 0
            and not (cleared := await self.__clear_focused_field()).success
        ):
            return cleared

        return await self.__type_characters(text=text)

    async def __type_characters(self, *, text: str) -> ActionResult:
        """
        Type text character-by-character to prevent ADB input drops.
        """

        last_result = ActionResult(success=True, duration=0)

        for character in text:
            escaped = self.__escape(text=character)

            if not (result := await self.__shell(command=f'input text "{escaped}"')).success:
                return result

            last_result = result
            await asyncio.sleep(0.01)

        return last_result

    async def __clear_focused_field(self) -> ActionResult:
        """
        Clear the focused Android text field using robust ADB key strategies.

        On SDK 30+ uses Ctrl+A (keycombination) to select all then deletes.
        Falls back to cursor-end + batched delete keyevents for older devices.
        Batching keycodes into a single ``adb shell input keyevent`` call is
        critical for performance (~100ms vs 5-8s for individual calls).
        """

        if (sdk_version := await self.__get_sdk_version()) >= AndroidClearStrategy.MODERN_MIN_SDK:
            select = await self.__shell(
                command=f"input keycombination {AndroidKeycode.CTRL_LEFT} {AndroidKeycode.A}"
            )
            if (
                select.success
                and (
                    delete := await self.__shell(
                        command=f"input keyevent {AndroidKeycode.DEL} {AndroidKeycode.DEL}"
                    )
                ).success
            ):
                return delete

            logger.warning("Modern clear failed (sdk=%d), falling back to legacy.", sdk_version)

        move_codes = f"{AndroidKeycode.MOVE_END} " + " ".join(
            [str(AndroidKeycode.DPAD_RIGHT)] * AndroidClearStrategy.RIGHT_ARROW_COUNT
        )
        if not (move := await self.__shell(command=f"input keyevent {move_codes}")).success:
            return move

        delete_codes = " ".join([str(AndroidKeycode.DEL)] * AndroidClearStrategy.DELETE_COUNT)
        return await self.__shell(command=f"input keyevent {delete_codes}")

    async def __get_sdk_version(self) -> int:
        """
        Return Android SDK version, or 0 when unavailable.
        """

        result = await self.__shell(command="getprop ro.build.version.sdk", capture_output=True)

        if not result.success or not result.output:
            return 0

        try:
            return int(str(result.output).strip())
        except ValueError:
            return 0

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
            logger.warning("Ignoring swipe speed for ADB adapter: %s", speed)

        duration = duration or (
            self.__configuration.interaction.policy.swipe.duration if self.__configuration else 300
        )

        return await self.__shell(command=f"input touchscreen swipe {x1} {y1} {x2} {y2} {duration}")

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
                await self.__abandon_unkillable_subprocess(process=process, arguments=arguments)

    async def __abandon_unkillable_subprocess(
        self,
        *,
        process: asyncio.subprocess.Process,
        arguments: List[str],
    ) -> None:
        """
        Bounded post-kill reap of a still-running subprocess.

        ``__run_safe_subprocess`` enforces a wall-clock per-command timeout
        via :func:`asyncio.wait_for`. Its ``finally`` block sends SIGKILL,
        then awaits the process to reap. When the host has wedged IO
        (emulator qcow2 backing exhausted, NFS hang, etc.) the kernel
        cannot deliver SIGKILL because the process sits in an
        uninterruptible-IO state; the await would block indefinitely and
        the entire workflow would stall.

        This helper caps that reap with ``subprocess_cleanup_timeout`` and
        abandons the process (logging at WARNING) rather than waiting
        forever. The leaked subprocess will be reaped by the OS once the
        underlying IO unwedges.
        """

        try:
            process.kill()
        except ProcessLookupError:
            return
        except Exception as exception:
            logger.warning(
                "Failed to send SIGKILL to subprocess",
                extra={
                    "component": "adapter.device.local.adb",
                    "event": "adb.subprocess.kill.failed",
                    "command": " ".join(arguments),
                    "error.message": str(exception),
                },
            )
            return

        try:
            await asyncio.wait_for(
                process.wait(),
                timeout=self.__configuration.subprocess_cleanup_timeout,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Subprocess cleanup wait timed out; abandoning process",
                extra={
                    "component": "adapter.device.local.adb",
                    "event": "adb.subprocess.cleanup.abandoned",
                    "command": " ".join(arguments),
                    "cleanup.budget": self.__configuration.subprocess_cleanup_timeout,
                    "process.pid": process.pid,
                },
            )
        except Exception as exception:
            logger.warning(
                "Failed to cleanup subprocess after kill",
                extra={
                    "component": "adapter.device.local.adb",
                    "event": "adb.subprocess.cleanup.failed",
                    "command": " ".join(arguments),
                    "error.message": str(exception),
                },
            )

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
                logger.warning("wait-for-device produced subprocess output unexpectedly")
            return returncode == 0
        except Exception:
            return False

    async def detect_keyboard(self) -> KeyboardObservation:
        """
        Detect soft-keyboard state via ``dumpsys`` and parse the touch-absorbing rectangle.
        """

        try:
            shown = await self.__keyboard_shown()
            if shown is None:
                return KeyboardObservation(visibility=KeyboardVisibility.UNKNOWN)
            if not shown:
                return KeyboardObservation(visibility=KeyboardVisibility.HIDDEN)
            bounds = await self.__keyboard_bounds()
            return KeyboardObservation(visibility=KeyboardVisibility.VISIBLE, bounds=bounds)
        except Exception as exception:
            logger.warning(f"ADB detect_keyboard failed: {exception}")
            return KeyboardObservation(visibility=KeyboardVisibility.UNKNOWN)

    async def __keyboard_shown(self) -> Optional[bool]:
        """
        Parse ``mInputShown=`` from ``dumpsys input_method``; None when the command fails.
        """

        result = await self.__shell(command="dumpsys input_method", capture_output=True)
        if not result.success or not result.output:
            return None
        match = re.search(r"mInputShown=(true|false)", result.output)
        if match is None:
            return None
        return match.group(1) == "true"

    async def __keyboard_bounds(self) -> Optional[Bounds]:
        """
        Parse the touch-absorbing rectangle from ``dumpsys window InputMethod``.
        """

        result = await self.__shell(command="dumpsys window InputMethod", capture_output=True)
        if not result.success or not result.output:
            return None
        match = re.search(
            r"touchable region=SkRegion\(\((\d+),(\d+),(\d+),(\d+)\)\)",
            result.output,
        )
        if match is None:
            return None
        left, top, right, bottom = (int(group) for group in match.groups())
        width = max(0, right - left)
        height = max(0, bottom - top)
        if width == 0 or height == 0:
            return None
        return Bounds(
            x=left,
            y=top,
            width=width,
            height=height,
            coordinate_system=CoordinateSystem.DEVICE_PIXEL,
        )

    async def dump_hierarchy(self) -> Optional[str]:
        """
        Dump UI hierarchy to XML string.

        Acquires the per-adapter UiAutomation lock with a bounded timeout. A
        prior ``dump_hierarchy`` task that was cancelled before the
        ``async with`` exit could leak the lock; treating the acquire as
        timed converts that latent class of bugs into an explicit
        :class:`DeviceError` instead of an indefinite await on the next call.
        """

        try:
            await asyncio.wait_for(
                self.__hierarchy_lock.acquire(),
                timeout=self.__configuration.hierarchy_lock_timeout,
            )
        except asyncio.TimeoutError as exception:
            raise DeviceError(
                f"Dump hierarchy: lock acquire timed out after "
                f"{self.__configuration.hierarchy_lock_timeout:.1f}s "
                "(likely a leaked lock from a prior cancelled task)"
            ) from exception

        try:
            return await self.__dump_hierarchy_locked()

        finally:
            self.__hierarchy_lock.release()

    async def __dump_hierarchy_locked(self) -> Optional[str]:
        """
        Dump UI hierarchy while holding the per-adapter UiAutomation lock.
        """

        path = ANDROID_UIAUTOMATION_DUMP_PATH

        await self.__recover_stale_ui_automation(reason="pre_dump")

        # Ensure we don't read a stale file
        await self.__shell(command=f"rm -f {path}")

        dump_result = await self.__run_uiautomator_dump(path=path, compressed=True)

        if not dump_result.success:
            if self.__uiautomator_timed_out(result=dump_result):
                raise DeviceError(
                    f"Dump hierarchy: compressed UI automation dump timed out: {dump_result.error or 'Unknown error'}"
                )

            logger.warning(
                f"Compressed dump failed: {dump_result.error}. Attempting recovery and fallback."
            )
            await self.__recover_stale_ui_automation(reason="compressed_dump_failed")
            dump_result = await self.__run_uiautomator_dump(path=path, compressed=False)

            if not dump_result.success:
                await self.__recover_stale_ui_automation(reason="fallback_dump_failed")
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

    async def __run_uiautomator_dump(self, *, path: str, compressed: bool) -> ActionResult:
        """
        Run one uiautomator dump attempt.
        """

        compression = " --compressed" if compressed else ""
        return await self.__shell(command=f"uiautomator dump{compression} {path}")

    @staticmethod
    def __uiautomator_timed_out(*, result: ActionResult) -> bool:
        """
        Return whether a dump result failed because the device command timed out.
        """

        return bool(result.error and ANDROID_UIAUTOMATION_TIMEOUT_MARKER in result.error.lower())

    async def __recover_stale_ui_automation(self, *, reason: str) -> None:
        """
        Release stale UiAutomation holders left by shell instrumentation or failed dumps.

        Android exposes only one UiAutomation registration at a time. A
        previous ``am instrument`` process can hold that slot forever,
        causing every later ``uiautomator dump`` to crash with
        "UiAutomationService ... already registered". Kill only shell
        ``app_process`` commands that are known UiAutomation holders.
        """

        state = await self.__shell(command="dumpsys accessibility", capture_output=True)
        if (
            not state.success
            or not state.output
            or ANDROID_UIAUTOMATION_ACTIVE_MARKER not in state.output
        ):
            return

        logger.warning(
            "Active UiAutomation registration detected before hierarchy dump; recovering.",
            extra={
                "component": "adapter.device.local.adb",
                "event": "adb.uiautomation.recovery.started",
                "reason": reason,
            },
        )

        cleanup = await self.__shell(command=self.__ui_automation_cleanup_command())
        if not cleanup.success:
            logger.warning(
                "UiAutomation recovery command failed: %s",
                cleanup.error,
                extra={
                    "component": "adapter.device.local.adb",
                    "event": "adb.uiautomation.recovery.failed",
                    "reason": reason,
                },
            )

        await asyncio.sleep(0.2)

    @staticmethod
    def __ui_automation_cleanup_command() -> str:
        """
        Return a shell command that kills only known stale UiAutomation holders.
        """

        return (
            f"for pid in $(pidof {ANDROID_UIAUTOMATION_PROCESS_NAME}); do "
            'cmdline=$(tr "\\0" " " < /proc/$pid/cmdline 2>/dev/null); '
            f'case "$cmdline" in '
            f'*"{ANDROID_UIAUTOMATION_INSTRUMENTATION_MARKER}"*|'
            f'*"{ANDROID_UIAUTOMATION_UIAUTOMATOR_MARKER}"*) '
            'kill -9 "$pid";; '
            "esac; "
            "done"
        )

    async def get_snapshot(self) -> Tuple[bytes, Optional[str]]:
        """
        Capture a required screenshot and best-effort XML hierarchy.
        """

        try:
            image = await self.capture_screen()
        except Exception as exception:
            logger.exception("Snapshot: Screenshot capture failed")
            raise DeviceError(f"Snapshot capture failed: {exception}") from exception

        if not image:
            raise DeviceError("Snapshot capture failed: empty screenshot")

        try:
            xml = await asyncio.wait_for(
                self.dump_hierarchy(),
                timeout=self.__configuration.snapshot_timeout,
            )
        except asyncio.TimeoutError:
            logger.exception(
                "Snapshot: Hierarchy dump timed out after %.1fs",
                self.__configuration.snapshot_timeout,
            )
            xml = None
        except Exception:
            logger.exception("Snapshot: Hierarchy dump failed")
            xml = None

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
                error_message = (
                    stderr.decode().strip()
                    if stderr
                    else f"ADB shell command exited with code {returncode}"
                )
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
