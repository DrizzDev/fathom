from __future__ import annotations

import asyncio
import io
import json
import re
import time
import xml.etree.ElementTree as ElementTree  # nosec
from logging import getLogger
from typing import Dict, List, Optional, Tuple

from PIL import Image

from fathom.adapters.ios.gateway import IOSAutomationGateway
from fathom.constants.interaction import SwipeSpeed
from fathom.constants.ios import IOSAdapterDefaults, IOSGestureDefaults
from fathom.constants.platform import DevicePlatform, IOSAutomationBackend
from fathom.core.exceptions import DeviceError
from fathom.interfaces.device import DevicePort
from fathom.schemas.configuration import (
    DeviceRuntimeConfiguration,
    InteractionPolicyConfiguration,
    InteractionRuntimeConfiguration,
    IOSConfiguration,
    ScrollInteractionPolicy,
    SwipeInteractionPolicy,
)
from fathom.schemas.results import ActionResult

logger = getLogger(__name__)


class IOSDevice(DevicePort):
    """
    Native iOS simulator adapter backed by xcrun simctl commands.
    """

    def __init__(
        self,
        *,
        configuration: Optional[IOSConfiguration] = None,
    ) -> None:
        """
        Initialize native iOS adapter configuration.
        """

        self.__configuration = configuration or IOSConfiguration()

        self.__cached_dimensions: Optional[Tuple[int, int]] = None
        self.__cached_automation_dimensions: Optional[Tuple[int, int]] = None

        self.__adapter_defaults = IOSAdapterDefaults()
        self.__gesture_defaults = IOSGestureDefaults()
        self.__automation_gateway = IOSAutomationGateway(configuration=self.__configuration)

        self.__runtime_configuration = DeviceRuntimeConfiguration(
            platform=DevicePlatform.IOS,
            identifier=self.__configuration.device_identifier,
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
            metadata={
                "backend": self.__configuration.automation_backend.value,
                "interaction_backend": (
                    "AUTOMATION_GATEWAY"
                    if self.__configuration.automation_backend
                    in {IOSAutomationBackend.XCUITEST, IOSAutomationBackend.WEBDRIVER_AGENT}
                    else "SIMCTL"
                ),
                "executable_path": self.__configuration.executable_path,
                "web_driver_agent_url": self.__configuration.web_driver_agent_url,
            },
        )

    @property
    def configuration(self) -> DeviceRuntimeConfiguration:
        """
        Return platform-neutral runtime configuration.
        """

        return self.__runtime_configuration

    async def tap(self, *, x: int, y: int) -> ActionResult:
        """
        Tap a screen point using the active iOS interaction backend.
        """

        start_time = time.time()

        try:
            if self.__should_route_interactions_via_automation_gateway():
                automation_x, automation_y = await self.__to_automation_coordinates(x=x, y=y)
                await self.__automation_gateway.tap(x=automation_x, y=automation_y)
            else:
                device_identifier = await self.__resolve_device_identifier()
                await self.__run_simctl(
                    parts=[
                        "io",
                        device_identifier,
                        "touchscreen",
                        "tap",
                        str(round(x)),
                        str(round(y)),
                    ],
                    timeout=self.__configuration.command_timeout,
                )
            return ActionResult(success=True, duration=int((time.time() - start_time) * 1000))
        except Exception as exception:
            return ActionResult(
                success=False,
                error=str(exception),
                duration=int((time.time() - start_time) * 1000),
            )

    async def type(self, *, text: str) -> ActionResult:
        """
        Type text with the active iOS interaction backend.
        """

        start_time = time.time()

        try:
            if self.__should_route_interactions_via_automation_gateway():
                await self.__automation_gateway.type_text(text=text)
            else:
                device_identifier = await self.__resolve_device_identifier()
                await self.__run_simctl(
                    parts=[
                        "io",
                        device_identifier,
                        "keyboard",
                        "type",
                        text,
                    ],
                    timeout=self.__configuration.command_timeout,
                )
            return ActionResult(success=True, duration=int((time.time() - start_time) * 1000))
        except Exception as exception:
            return ActionResult(
                success=False,
                error=str(exception),
                duration=int((time.time() - start_time) * 1000),
            )

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
        Swipe between two points using the active iOS interaction backend.
        """

        start_time = time.time()

        requested_speed = speed
        resolved_duration = duration or (
            self.__configuration.interaction.policy.swipe.duration_milliseconds
        )

        if requested_speed is not None:
            logger.debug("Ignoring swipe speed for iOS simctl adapter: %s", requested_speed)

        try:
            if self.__should_route_interactions_via_automation_gateway():
                start_x, start_y = await self.__to_automation_coordinates(x=x1, y=y1)
                end_x, end_y = await self.__to_automation_coordinates(x=x2, y=y2)
                await self.__automation_gateway.swipe(
                    start_x=start_x,
                    start_y=start_y,
                    end_x=end_x,
                    end_y=end_y,
                    duration_milliseconds=resolved_duration,
                )
            else:
                device_identifier = await self.__resolve_device_identifier()
                await self.__run_simctl(
                    parts=[
                        "io",
                        device_identifier,
                        "touchscreen",
                        "swipe",
                        str(round(x1)),
                        str(round(y1)),
                        str(round(x2)),
                        str(round(y2)),
                        self.__format_swipe_seconds(duration_ms=resolved_duration),
                    ],
                    timeout=self.__configuration.command_timeout,
                )
            return ActionResult(success=True, duration=int((time.time() - start_time) * 1000))
        except Exception as exception:
            return ActionResult(
                success=False,
                error=str(exception),
                duration=int((time.time() - start_time) * 1000),
            )

    async def back(self) -> ActionResult:
        """
        Attempt back navigation via a left-edge swipe gesture.
        """

        width, height = await self.get_dimensions()
        y = int(height * self.__gesture_defaults.back_y_ratio)
        x1 = int(width * self.__gesture_defaults.back_start_x_ratio)
        x2 = int(width * self.__gesture_defaults.back_end_x_ratio)

        return await self.swipe(
            x1=x1,
            y1=y,
            x2=x2,
            y2=y,
            duration=self.__gesture_defaults.back_duration_milliseconds,
        )

    async def home(self) -> ActionResult:
        """
        Navigate to the iOS home screen by launching SpringBoard.
        """

        start_time = time.time()

        try:
            if self.__should_route_interactions_via_automation_gateway():
                await self.__automation_gateway.press_home()
            else:
                device_identifier = await self.__resolve_device_identifier()
                await self.__run_simctl(
                    parts=[
                        "launch",
                        device_identifier,
                        self.__adapter_defaults.springboard_bundle_identifier,
                    ],
                    timeout=self.__configuration.command_timeout,
                )
            return ActionResult(success=True, duration=int((time.time() - start_time) * 1000))
        except Exception as exception:
            return ActionResult(
                success=False,
                error=str(exception),
                duration=int((time.time() - start_time) * 1000),
            )

    async def get_dimensions(self) -> Tuple[int, int]:
        """
        Get current simulator dimensions by parsing screenshot PNG headers.
        """

        if self.__cached_dimensions:
            return self.__cached_dimensions

        screenshot = await self.capture_screen()

        try:
            width, height = self.__parse_png_dimensions(image=screenshot)
        except DeviceError:
            width, height = self.__parse_image_dimensions(image=screenshot)

        if width <= 0 or height <= 0:
            raise DeviceError("Get dimensions: invalid dimensions returned by simulator")

        self.__cached_dimensions = (width, height)
        return self.__cached_dimensions

    async def capture_screen(self) -> bytes:
        """
        Capture screenshot bytes from simctl.
        """

        device_identifier = await self.__resolve_device_identifier()

        return_code, stdout, stderr = await self.__run_simctl(
            parts=["io", device_identifier, "screenshot", "--type=png", "-"],
            timeout=self.__configuration.command_timeout,
            capture_stdout=True,
            capture_stderr=True,
        )

        if return_code != 0:
            error_message = (
                stderr.decode("utf-8", errors="ignore").strip() if stderr else "Unknown error"
            )
            raise DeviceError(f"Capture screen: simctl screenshot failed: {error_message}")

        if len(stdout) < self.__adapter_defaults.screenshot_minimum_bytes:
            raise DeviceError("Capture screen: screenshot payload was empty or truncated")

        self.__cache_dimensions_from_image(image=stdout)
        return stdout

    async def dump_hierarchy(self) -> Optional[str]:
        """
        Capture hierarchy XML from the configured iOS automation backend.
        """

        if not self.__should_route_interactions_via_automation_gateway():
            raise DeviceError(
                "xcrun simctl does not currently expose a native XML hierarchy dump. "
                "Use XCUITEST/WEBDRIVER_AGENT backend for hierarchy extraction."
            )

        try:
            return await self.__automation_gateway.dump_source()
        except DeviceError as exception:
            raise DeviceError(
                f"Dump hierarchy: failed to fetch iOS hierarchy XML: {exception}"
            ) from exception

    async def get_current_package(self) -> str:
        """
        Resolve foreground bundle identifier using automation, XML fallback, then launchctl.
        """

        configured_bundle_identifier = self.__configuration.bundle_identifier

        automation_bundle_identifier = await self.__resolve_automation_bundle_identifier()
        if automation_bundle_identifier:
            return automation_bundle_identifier

        return await self.__resolve_launchctl_bundle_identifier(
            configured_bundle_identifier=configured_bundle_identifier
        )

    async def get_snapshot(self) -> Tuple[bytes, Optional[str]]:
        """
        Capture iOS screenshot and hierarchy together.
        """

        screenshot_result, hierarchy_result = await asyncio.gather(
            self.capture_screen(),
            self.dump_hierarchy(),
        )

        return screenshot_result, hierarchy_result

    def __cache_dimensions_from_image(self, *, image: bytes) -> None:
        """
        Cache screenshot pixel dimensions from PNG bytes when available.
        """

        if self.__cached_dimensions:
            return

        try:
            width, height = self.__parse_png_dimensions(image=image)
        except DeviceError:
            width, height = self.__parse_image_dimensions(image=image)

        if width > 0 and height > 0:
            self.__cached_dimensions = (width, height)

    async def __resolve_launchctl_bundle_identifier(
        self,
        *,
        configured_bundle_identifier: str | None,
    ) -> str:
        """
        Resolve the foreground bundle identifier using simulator launchctl output.
        """

        device_identifier = await self.__resolve_device_identifier()

        try:
            return_code, stdout, stderr = await self.__run_simctl(
                parts=["spawn", device_identifier, "launchctl", "list"],
                timeout=self.__configuration.command_timeout,
                capture_stdout=True,
                capture_stderr=True,
            )
        except Exception as exception:
            logger.warning("Get current package: launchctl invocation failed: %s", exception)
            return configured_bundle_identifier or self.__adapter_defaults.unknown_bundle_identifier

        if return_code != 0:
            error_message = (
                stderr.decode("utf-8", errors="ignore").strip() if stderr else "Unknown error"
            )
            logger.warning(
                "Get current package: launchctl returned non-zero status: %s", error_message
            )
            return configured_bundle_identifier or self.__adapter_defaults.unknown_bundle_identifier

        launchctl_output = stdout.decode("utf-8", errors="ignore")
        bundle_identifier = self.__extract_foreground_bundle_identifier(
            launchctl_output=launchctl_output
        )

        if bundle_identifier:
            return bundle_identifier

        return configured_bundle_identifier or self.__adapter_defaults.unknown_bundle_identifier

    async def __resolve_automation_bundle_identifier(
        self,
    ) -> str | None:
        """
        Resolve the foreground bundle identifier through automation-backed sources.
        """

        if not self.__should_route_interactions_via_automation_gateway():
            return None

        bundle_identifier = await self.__resolve_active_application_bundle_identifier()
        if bundle_identifier:
            return bundle_identifier

        return await self.__resolve_gateway_hierarchy_bundle_identifier()

    async def __resolve_active_application_bundle_identifier(self) -> str | None:
        """
        Resolve the foreground bundle identifier through WebDriverAgent active app info.
        """

        try:
            return await self.__automation_gateway.get_active_application_bundle_identifier()
        except DeviceError as exception:
            logger.warning(
                "Get current package: active application lookup failed: %s",
                exception,
            )
            return None

    async def __resolve_gateway_hierarchy_bundle_identifier(self) -> str | None:
        """
        Resolve the foreground bundle identifier from WebDriverAgent hierarchy XML.
        """

        try:
            hierarchy_content = await self.__automation_gateway.dump_source()
        except DeviceError as exception:
            logger.warning(
                "Get current package: hierarchy fallback lookup failed: %s",
                exception,
            )
            return None

        return self.__extract_bundle_identifier_from_hierarchy(hierarchy_content=hierarchy_content)

    def __extract_bundle_identifier_from_hierarchy(
        self,
        *,
        hierarchy_content: str | None,
    ) -> str | None:
        """
        Extract the foreground bundle identifier from iOS hierarchy XML when available.
        """

        if not hierarchy_content:
            return None

        try:
            root = ElementTree.fromstring(hierarchy_content)  # nosec
        except Exception as exception:
            logger.warning(
                "Get current package: failed to parse hierarchy XML fallback: %s",
                exception,
            )
            return None

        if root.tag == "XCUIElementTypeApplication":
            bundle_identifier = root.attrib.get("bundleId")
            if isinstance(bundle_identifier, str) and bundle_identifier.strip():
                return bundle_identifier.strip()

        application_node = root.find(".//*[@bundleId]")
        if application_node is None:
            return None

        bundle_identifier = application_node.attrib.get("bundleId")
        if not isinstance(bundle_identifier, str) or not bundle_identifier.strip():
            return None

        return bundle_identifier.strip()

    async def wait_for_device(self, *, timeout: float) -> bool:
        """
        Wait for a booted simulator and valid screen dimensions.
        """

        start_time = time.time()

        while (time.time() - start_time) < timeout:
            try:
                await self.__resolve_device_identifier()
                await self.get_dimensions()
                return True
            except Exception as exception:
                logger.debug("Device readiness check pending: %s", exception)
                await asyncio.sleep(self.__adapter_defaults.device_ready_poll_seconds)

        return False

    async def close(self) -> None:
        """
        Close adapter resources.
        """

        return None

    def __should_route_interactions_via_automation_gateway(self) -> bool:
        """
        Determine whether gestures should use the automation gateway backend.
        """

        return self.__configuration.automation_backend in {
            IOSAutomationBackend.XCUITEST,
            IOSAutomationBackend.WEBDRIVER_AGENT,
        }

    async def __to_automation_coordinates(self, *, x: int, y: int) -> Tuple[float, float]:
        """
        Convert screenshot pixel coordinates into automation-window coordinates.
        """

        screenshot_width, screenshot_height = await self.get_dimensions()
        automation_width, automation_height = await self.__get_automation_window_size()

        if screenshot_width <= 0 or screenshot_height <= 0:
            raise DeviceError("Invalid iOS screenshot dimensions for coordinate conversion")

        return (
            float(x) * float(automation_width) / float(screenshot_width),
            float(y) * float(automation_height) / float(screenshot_height),
        )

    async def __get_automation_window_size(self) -> Tuple[int, int]:
        """
        Resolve and cache automation-window dimensions in logical points.
        """

        if self.__cached_automation_dimensions:
            return self.__cached_automation_dimensions

        window_size = await self.__automation_gateway.get_window_size()
        self.__cached_automation_dimensions = window_size
        return window_size

    async def __resolve_device_identifier(self) -> str:
        """
        Resolve configured simulator identifier or select the first booted simulator.
        """

        if self.__configuration.device_identifier:
            return self.__configuration.device_identifier

        device_identifier = await self.__select_booted_device_identifier()

        self.__configuration.device_identifier = device_identifier
        self.__runtime_configuration.identifier = device_identifier

        return device_identifier

    def __parse_simulator_inventory_payload(self, *, stdout: bytes) -> Dict[str, object]:
        """
        Parse simctl JSON payload and return validated devices map.
        """

        try:
            payload = json.loads(stdout.decode("utf-8", errors="ignore"))
        except Exception as exception:
            raise DeviceError(
                f"Failed to parse simulator inventory JSON: {exception}"
            ) from exception

        devices = payload.get("devices", {})
        if not isinstance(devices, dict):
            raise DeviceError("Simulator inventory response did not contain a devices map")

        return devices

    def __resolve_booted_identifier_from_runtime_devices(
        self, *, runtime_devices: object
    ) -> Optional[str]:
        """
        Resolve a booted device identifier from runtime-scoped simulator entries.
        """

        if not isinstance(runtime_devices, list):
            return None

        for device in runtime_devices:
            if not isinstance(device, dict):
                continue

            state = str(device.get("state", "")).strip().lower()
            is_available = device.get("isAvailable")
            if is_available is False:
                continue

            if state == "booted":
                discovered_device_identifier = str(device.get("udid", "")).strip()
                if discovered_device_identifier:
                    return discovered_device_identifier

        return None

    def __find_booted_device_identifier(self, *, devices: Dict[str, object]) -> Optional[str]:
        """
        Find first booted simulator identifier across runtimes.
        """

        for runtime_devices in devices.values():
            discovered_device_identifier = self.__resolve_booted_identifier_from_runtime_devices(
                runtime_devices=runtime_devices
            )
            if discovered_device_identifier:
                return discovered_device_identifier

        return None

    async def __select_booted_device_identifier(self) -> str:
        """
        Select the first available booted simulator from simctl inventory.
        """

        return_code, stdout, stderr = await self.__run_simctl(
            parts=["list", "devices", "--json"],
            timeout=self.__configuration.command_timeout,
            capture_stdout=True,
            capture_stderr=True,
        )

        if return_code != 0:
            error_message = (
                stderr.decode("utf-8", errors="ignore").strip() if stderr else "Unknown error"
            )
            raise DeviceError(f"Failed to list simulators: {error_message}")

        devices = self.__parse_simulator_inventory_payload(stdout=stdout)
        discovered_device_identifier = self.__find_booted_device_identifier(devices=devices)
        if discovered_device_identifier:
            return discovered_device_identifier

        raise DeviceError(
            "No booted iOS simulator found. Boot a simulator and/or pass explicit device identifier."
        )

    async def __run_simctl(
        self,
        *,
        parts: List[str],
        timeout: float,
        capture_stdout: bool = False,
        capture_stderr: bool = True,
    ) -> Tuple[int, bytes, bytes]:
        """
        Run an xcrun simctl command and return process outputs.
        """

        arguments = [
            self.__configuration.executable_path,
            self.__adapter_defaults.simulator_control_command,
            *parts,
        ]

        return await self.__run_safe_subprocess(
            timeout=timeout,
            arguments=arguments,
            capture_stdout=capture_stdout,
            capture_stderr=capture_stderr,
        )

    async def __run_safe_subprocess(
        self,
        *,
        timeout: float,
        arguments: List[str],
        capture_stdout: bool,
        capture_stderr: bool,
    ) -> Tuple[int, bytes, bytes]:
        """
        Execute subprocess safely and ensure process cleanup on failure.
        """

        process: Optional[asyncio.subprocess.Process] = None

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
                    logger.debug("Subprocess already terminated before cleanup completed.")
                except Exception as cleanup_exception:
                    logger.warning("Failed to cleanup subprocess: %s", cleanup_exception)

    @staticmethod
    def __format_swipe_seconds(*, duration_ms: int) -> str:
        """
        Convert swipe duration from milliseconds to simctl seconds format.
        """

        seconds = max(0.0, float(duration_ms) / 1000.0)
        formatted = f"{seconds:.3f}"

        return formatted.rstrip("0").rstrip(".") or "0"

    @staticmethod
    def __parse_png_dimensions(*, image: bytes) -> Tuple[int, int]:
        """
        Parse width and height from PNG IHDR bytes.
        """

        if len(image) < 24:
            raise DeviceError("Invalid PNG payload: too small to contain IHDR")

        signature = image[:8]
        if signature != b"\x89PNG\r\n\x1a\n":
            raise DeviceError("Invalid PNG payload: missing PNG signature")

        chunk_type = image[12:16]
        if chunk_type != b"IHDR":
            raise DeviceError("Invalid PNG payload: IHDR chunk not found")

        width = int.from_bytes(image[16:20], byteorder="big")
        height = int.from_bytes(image[20:24], byteorder="big")

        return width, height

    @staticmethod
    def __parse_image_dimensions(*, image: bytes) -> Tuple[int, int]:
        """
        Parse screenshot width and height using PIL. Handles any image format.
        """

        try:
            with Image.open(io.BytesIO(image)) as img:
                width, height = img.size
                return width, height
        except Exception as exception:
            raise DeviceError(f"Unable to determine image dimensions: {exception}") from exception

    def __extract_foreground_bundle_identifier(self, *, launchctl_output: str) -> Optional[str]:
        """
        Extract foreground bundle identifier from launchctl lines.
        """

        candidates: List[str] = []
        pattern = re.compile(r"UIKitApplication:([^\[\s]+)")

        for line in launchctl_output.splitlines():
            match = pattern.search(line)
            if not match:
                continue

            if candidate := match.group(1).strip():
                candidates.append(candidate)

        for candidate in candidates:
            if candidate != self.__adapter_defaults.springboard_bundle_identifier:
                return candidate

        if candidates:
            return candidates[0]

        return None
