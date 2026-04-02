from __future__ import annotations

import time
from logging import getLogger
from typing import Optional, Tuple

from fathom.core.exceptions import DeviceError
from fathom.interfaces.device import DevicePort
from fathom.interfaces.perception import PerceptionPort
from fathom.schemas.configuration import DeviceRuntimeConfiguration
from fathom.schemas.screens import ScreenCapture

logger = getLogger(__name__)


class RemotePerceptionAdapter(PerceptionPort):
    """
    Remote perception adapter backed by the remote device implementation.
    """

    def __init__(self, *, device: DevicePort, include_hierarchy: bool) -> None:
        """
        Initialize the remote perception adapter.
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
        Capture remote screenshot and optional hierarchy in a single snapshot.
        """

        capture_start = time.time()

        if self.__include_hierarchy:
            screenshot_bytes, hierarchy_content = await self.__capture()
        else:
            hierarchy_content = None
            screenshot_bytes = await self.__device.capture_screen()

        if not screenshot_bytes:
            raise DeviceError("Remote perception captured an empty screenshot.")

        width, height = await self.__device.get_dimensions()
        application_identifier = await self.__device.get_current_package()

        return ScreenCapture(
            width=width,
            height=height,
            image=screenshot_bytes,
            xml_content=hierarchy_content,
            activity=application_identifier,
            timestamp=int(time.time() * 1000),
            metadata={
                "capture_duration": time.time() - capture_start,
                **(
                    {"hierarchy_dump_duration": time.time() - capture_start}
                    if hierarchy_content is not None
                    else {}
                ),
            },
        )

    async def __capture(self) -> Tuple[bytes, Optional[str]]:
        """
        Capture a screenshot and best-effort hierarchy using the current snapshot contract.
        """

        try:
            screenshot_bytes, hierarchy_content = await self.__device.get_snapshot()
            if screenshot_bytes:
                return screenshot_bytes, hierarchy_content

            logger.warning(
                "Remote snapshot returned an empty screenshot. Falling back to separate capture calls."
            )
        except DeviceError as exception:
            if not exception.retryable:
                raise

            logger.warning(
                "Remote snapshot failed with retryable error. Falling back to separate capture calls: %s",
                exception,
            )

        return await self.__capture_with_separate_calls()

    async def __capture_with_separate_calls(self) -> Tuple[bytes, Optional[str]]:
        """
        Capture the required screenshot first, then try to fetch hierarchy best-effort.
        """

        screenshot_bytes = await self.__device.capture_screen()

        try:
            hierarchy_content = await self.__device.dump_hierarchy()
        except DeviceError as exception:
            hierarchy_content = None
            logger.warning("Remote hierarchy fallback failed: %s", exception)

        return screenshot_bytes, hierarchy_content
