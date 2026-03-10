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

    async def capture(self) -> ScreenCapture:
        """
        Capture iOS screenshot without hierarchy enhancement.
        """

        screenshot_bytes = await self.__device.capture_screen()
        if not screenshot_bytes:
            raise DeviceError("iOS native perception captured an empty screenshot.")

        width, height = await self.__device.get_dimensions()
        application_identifier = await self.__device.get_current_package()

        return ScreenCapture(
            width=width,
            height=height,
            activity=application_identifier,
            image=screenshot_bytes,
            timestamp=int(time.time() * 1000),
            metadata={"perception_strategy": "native"},
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

    async def capture(self) -> ScreenCapture:
        """
        Capture iOS screenshot with optional hierarchy enhancement.
        """

        screenshot_bytes, hierarchy_content = await self.__device.get_snapshot()

        if not screenshot_bytes:
            raise DeviceError("iOS enhanced perception captured an empty screenshot.")

        width, height = await self.__device.get_dimensions()
        application_identifier = await self.__device.get_current_package()

        metadata = {"perception_strategy": "enhanced"}
        if hierarchy_content is None:
            metadata["hierarchy_error"] = "Hierarchy unavailable"

        return ScreenCapture(
            width=width,
            height=height,
            activity=application_identifier,
            image=screenshot_bytes,
            xml_content=hierarchy_content,
            timestamp=int(time.time() * 1000),
            metadata=metadata,
        )
