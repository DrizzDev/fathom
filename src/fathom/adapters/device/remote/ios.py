from __future__ import annotations

from typing import Optional, Tuple

from fathom.constants.interaction import SwipeSpeed
from fathom.core.exceptions import DeviceError
from fathom.interfaces.device import DevicePort
from fathom.schemas.configuration import DeviceConfiguration, DeviceRuntimeConfiguration
from fathom.schemas.results import ActionResult
from fathom.utils.image import parse_png_dimensions

from .adb import ADBRemoteDeviceAdapter


class IOSRemoteDeviceAdapter(DevicePort):
    """
    Remote iOS adapter that normalizes screenshot-space coordinates before transport.
    """

    def __init__(
        self,
        configuration: DeviceConfiguration,
        *,
        delegate: Optional[ADBRemoteDeviceAdapter] = None,
    ) -> None:
        """
        Initialize the remote iOS adapter.
        """

        self.__delegate = delegate or ADBRemoteDeviceAdapter(configuration=configuration)

        self.__cached_screenshot_dimensions: Optional[Tuple[int, int]] = None
        self.__cached_automation_dimensions: Optional[Tuple[int, int]] = None

    @property
    def configuration(self) -> Optional[DeviceRuntimeConfiguration]:
        """
        Return platform-neutral device configuration.
        """

        return self.__delegate.configuration

    async def tap(self, *, x: int, y: int) -> ActionResult:
        """
        Tap after converting screenshot-space coordinates into iOS automation coordinates.
        """

        automation_x, automation_y = await self.__to_automation_coordinates(x=x, y=y)
        return await self.__delegate.tap(x=automation_x, y=automation_y)

    async def type(self, *, text: str) -> ActionResult:
        """
        Delegate remote text entry.
        """

        return await self.__delegate.type(text=text)

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
        Swipe after converting screenshot-space coordinates into iOS automation coordinates.
        """

        start_x, start_y = await self.__to_automation_coordinates(x=x1, y=y1)
        end_x, end_y = await self.__to_automation_coordinates(x=x2, y=y2)

        return await self.__delegate.swipe(
            x1=start_x,
            y1=start_y,
            x2=end_x,
            y2=end_y,
            speed=speed,
            duration=duration,
        )

    async def back(self) -> ActionResult:
        """
        Delegate remote back action.
        """

        return await self.__delegate.back()

    async def home(self) -> ActionResult:
        """
        Delegate remote home action.
        """

        return await self.__delegate.home()

    async def get_current_package(self) -> str:
        """
        Delegate current package lookup.
        """

        return await self.__delegate.get_current_package()

    async def capture_screen(self) -> bytes:
        """
        Capture remote screenshot bytes and cache screenshot-space dimensions.
        """

        image = await self.__delegate.capture_screen()
        self.__cache_screenshot_dimensions(image=image)
        return image

    async def dump_hierarchy(self) -> Optional[str]:
        """
        Delegate remote hierarchy dump.
        """

        return await self.__delegate.dump_hierarchy()

    async def get_snapshot(self) -> Tuple[bytes, Optional[str]]:
        """
        Capture remote screenshot and hierarchy while caching screenshot-space dimensions.
        """

        image, hierarchy = await self.__delegate.get_snapshot()
        self.__cache_screenshot_dimensions(image=image)
        return image, hierarchy

    async def get_dimensions(self) -> Tuple[int, int]:
        """
        Return screenshot-space dimensions to match the local iOS adapter contract.
        """

        if self.__cached_screenshot_dimensions:
            return self.__cached_screenshot_dimensions

        image = await self.capture_screen()
        self.__cache_screenshot_dimensions(image=image)

        if not self.__cached_screenshot_dimensions:
            raise DeviceError("Get dimensions: remote iOS screenshot dimensions were unavailable")

        return self.__cached_screenshot_dimensions

    async def wait_for_device(self, *, timeout: float) -> bool:
        """
        Delegate remote device readiness checks.
        """

        return await self.__delegate.wait_for_device(timeout=timeout)

    async def close(self) -> None:
        """
        Close the underlying remote adapter client.
        """

        await self.__delegate.close()

    async def __to_automation_coordinates(self, *, x: int, y: int) -> Tuple[int, int]:
        """
        Convert screenshot-space pixels into remote iOS automation-window coordinates.
        """

        screenshot_width, screenshot_height = await self.get_dimensions()
        automation_width, automation_height = await self.__get_automation_dimensions()

        if screenshot_width <= 0 or screenshot_height <= 0:
            raise DeviceError("Invalid remote iOS screenshot dimensions for coordinate conversion")

        return (
            round(float(x) * float(automation_width) / float(screenshot_width)),
            round(float(y) * float(automation_height) / float(screenshot_height)),
        )

    async def __get_automation_dimensions(self) -> Tuple[int, int]:
        """
        Resolve logical iOS automation-window dimensions from the remote backend.
        """

        if self.__cached_automation_dimensions:
            return self.__cached_automation_dimensions

        self.__cached_automation_dimensions = await self.__delegate.get_dimensions()
        return self.__cached_automation_dimensions

    def __cache_screenshot_dimensions(self, *, image: bytes) -> None:
        """
        Cache screenshot-space dimensions from PNG bytes.
        """

        if self.__cached_screenshot_dimensions or not image:
            return

        try:
            self.__cached_screenshot_dimensions = parse_png_dimensions(image)
        except ValueError as exc:
            raise DeviceError(str(exc)) from exc
