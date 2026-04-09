from __future__ import annotations

import time
from logging import getLogger
from typing import Optional

from fathom.core.exceptions import DeviceError
from fathom.interfaces.device import DevicePort
from fathom.interfaces.perception import PerceptionPort
from fathom.schemas.configuration import DeviceRuntimeConfiguration
from fathom.schemas.screens import ScreenCapture
from fathom.utils.image import parse_png_dimensions

logger = getLogger(__name__)


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

        capture_start = time.time()
        if self.__include_hierarchy:
            screenshot_bytes, hierarchy_content = await self.__device.get_snapshot()
        else:
            screenshot_bytes = await self.__device.capture_screen()
            hierarchy_content = None

        if not screenshot_bytes:
            raise DeviceError("Android perception captured an empty screenshot.")

        width, height = await self.__resolve_dimensions(screenshot_bytes)

        try:
            application_identifier = await self.__device.get_current_package()
        except Exception:
            application_identifier = "unknown"

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

    async def __resolve_dimensions(self, screenshot_bytes: bytes) -> tuple[int, int]:
        """Derive authoritative screen dimensions from the screenshot PNG.

        Falls back to ``device.get_dimensions()`` only when the
        screenshot bytes cannot be parsed as PNG — e.g. raw framebuffer
        or JPEG payloads on non-standard devices.
        """

        try:
            return parse_png_dimensions(screenshot_bytes)
        except ValueError:
            logger.warning(
                "Could not parse PNG dimensions from screenshot; "
                "falling back to device.get_dimensions()"
            )
            return await self.__device.get_dimensions()
