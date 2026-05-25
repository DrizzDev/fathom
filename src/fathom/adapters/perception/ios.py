from __future__ import annotations

import time
from typing import Optional

from fathom.core.exceptions import DeviceError
from fathom.interfaces.device import DevicePort
from fathom.interfaces.perception import PerceptionPort
from fathom.schemas.configuration import DeviceRuntimeConfiguration
from fathom.schemas.screens import ScreenCapture


class IOSNativePerceptionAdapter(PerceptionPort):
    """
    Native iOS perception adapter with screenshot-only capture.
    """

    def __init__(self, *, device: DevicePort) -> None:
        """
        Initialize native iOS perception adapter.
        """

        self.__device = device

    @property
    def configuration(self) -> Optional[DeviceRuntimeConfiguration]:
        """
        Return platform-neutral runtime configuration.
        """

        return self.__device.configuration

    async def detect_keyboard(self, *, capture: Optional[ScreenCapture] = None):
        """
        Delegate keyboard detection to the underlying iOS device adapter (XCUITest XML walk).
        """

        _ = capture
        return await self.__device.detect_keyboard()

    async def capture(self) -> ScreenCapture:
        """
        Capture iOS screenshot without hierarchy enhancement.
        """

        capture_start = time.time()
        screenshot_bytes = await self.__device.capture_screen()
        if not screenshot_bytes:
            raise DeviceError("iOS native perception captured an empty screenshot.")

        width, height = await self.__device.get_dimensions()
        application_identifier = await self.__device.get_current_package()

        return ScreenCapture(
            width=width,
            height=height,
            image=screenshot_bytes,
            activity=application_identifier,
            timestamp=int(time.time() * 1000),
            metadata={
                "perception_strategy": "native",
                "capture_duration": time.time() - capture_start,
            },
        )


class IOSEnhancedPerceptionAdapter(PerceptionPort):
    """
    Enhanced iOS perception adapter with optional hierarchy extraction.
    """

    def __init__(self, *, device: DevicePort) -> None:
        """
        Initialize enhanced iOS perception adapter.
        """

        self.__device = device

    @property
    def configuration(self) -> Optional[DeviceRuntimeConfiguration]:
        """
        Return platform-neutral runtime configuration.
        """

        return self.__device.configuration

    async def detect_keyboard(self, *, capture: Optional[ScreenCapture] = None):
        """
        Delegate keyboard detection to the underlying iOS device adapter (XCUITest XML walk).
        """

        _ = capture
        return await self.__device.detect_keyboard()

    async def capture(self) -> ScreenCapture:
        """
        Capture iOS screenshot with optional hierarchy enhancement.
        """

        capture_start = time.time()
        screenshot_bytes, hierarchy_content = await self.__device.get_snapshot()

        if not screenshot_bytes:
            raise DeviceError("iOS enhanced perception captured an empty screenshot.")

        width, height = await self.__device.get_dimensions()
        application_identifier = await self.__device.get_current_package()

        metadata = {
            "perception_strategy": "enhanced",
            "capture_duration": time.time() - capture_start,
        }
        if hierarchy_content is None:
            metadata["hierarchy_error"] = "Hierarchy unavailable"
        else:
            metadata["hierarchy_dump_duration"] = time.time() - capture_start

        return ScreenCapture(
            width=width,
            height=height,
            metadata=metadata,
            image=screenshot_bytes,
            xml_content=hierarchy_content,
            activity=application_identifier,
            timestamp=int(time.time() * 1000),
        )
