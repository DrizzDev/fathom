from __future__ import annotations

import time
from typing import Optional

from fathom.core.exceptions import DeviceError
from fathom.interfaces.device import DevicePort
from fathom.interfaces.perception import PerceptionPort
from fathom.schemas.configuration import DeviceRuntimeConfiguration
from fathom.schemas.screens import ScreenCapture


class AndroidPerceptionAdapter(PerceptionPort):
    """
    Android perception adapter backed by the existing ADB device implementation.
    """

    def __init__(self, *, device: DevicePort, include_hierarchy: bool) -> None:
        """
        Initialize Android perception adapter.
        """

        self.__device = device
        self.__include_hierarchy = include_hierarchy

    @property
    def configuration(self) -> Optional[DeviceRuntimeConfiguration]:
        """
        Return platform-neutral runtime configuration.
        """

        return self.__device.configuration

    async def capture(self) -> ScreenCapture:
        """
        Capture Android screenshot and optional hierarchy in a single snapshot.
        """

        if self.__include_hierarchy:
            screenshot_bytes, hierarchy_content = await self.__device.get_snapshot()
        else:
            screenshot_bytes = await self.__device.capture_screen()
            hierarchy_content = None

        if not screenshot_bytes:
            raise DeviceError("Android perception captured an empty screenshot.")

        width, height = await self.__device.get_dimensions()
        application_identifier = await self.__device.get_current_package()

        return ScreenCapture(
            width=width,
            height=height,
            activity=application_identifier,
            image=screenshot_bytes,
            xml_content=hierarchy_content,
            timestamp=int(time.time() * 1000),
        )
