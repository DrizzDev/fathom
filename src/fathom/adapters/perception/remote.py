from __future__ import annotations

import time
from typing import Tuple

from fathom.adapters.device.remote import RemoteDeviceAdapter
from fathom.core.exceptions import DeviceError
from fathom.interfaces.perception import PerceptionPort
from fathom.schemas.configuration import DeviceRuntimeConfiguration
from fathom.schemas.screens import ScreenCapture


class RemotePerceptionAdapter(PerceptionPort):
    """
    Remote perception adapter backed by the remote device implementation.
    """

    def __init__(self, *, device: RemoteDeviceAdapter) -> None:
        """
        Initialize the remote perception adapter.
        """

        self.__device = device

    @property
    def configuration(self) -> DeviceRuntimeConfiguration:
        """
        Return platform-neutral runtime configuration.
        """

        return self.__device.configuration

    async def capture(self) -> ScreenCapture:
        """
        Capture remote screenshot and optional hierarchy in a single snapshot.
        """

        screenshot_bytes, hierarchy_content = await self.__device.get_snapshot()
        if not screenshot_bytes:
            raise DeviceError("Remote perception captured an empty screenshot.")

        width, height = await self.__device.get_dimensions()
        application_identifier = await self.get_current_application()

        return ScreenCapture(
            width=width,
            height=height,
            activity=application_identifier,
            image=screenshot_bytes,
            xml_content=hierarchy_content,
            timestamp=int(time.time() * 1000),
        )

    async def get_current_application(self) -> str:
        """
        Resolve current remote foreground application identifier.
        """

        return await self.__device.get_current_package()

    async def get_dimensions(self) -> Tuple[int, int]:
        """
        Return remote screen dimensions.
        """

        return await self.__device.get_dimensions()
