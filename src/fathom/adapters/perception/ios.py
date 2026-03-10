from __future__ import annotations

import asyncio
import time
import xml.etree.ElementTree as ElementTree  # nosec
from typing import Tuple, cast

from fathom.adapters.device.ios import IOSDevice
from fathom.core.exceptions import DeviceError
from fathom.interfaces.hierarchy import HierarchyPort
from fathom.interfaces.perception import PerceptionPort
from fathom.schemas.configuration import DeviceRuntimeConfiguration
from fathom.schemas.screens import ScreenCapture


class IOSNativePerceptionAdapter(PerceptionPort):
    """
    Native iOS perception adapter with screenshot-only capture.
    """

    def __init__(self, *, device: IOSDevice) -> None:
        """
        Initialize native iOS perception adapter.
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
        Capture iOS screenshot without hierarchy enhancement.
        """

        screenshot_bytes = await self.__device.capture_screen()
        if not screenshot_bytes:
            raise DeviceError("iOS native perception captured an empty screenshot.")

        width, height = await self.__device.get_dimensions()
        application_identifier = await self.get_current_application()

        return ScreenCapture(
            width=width,
            height=height,
            activity=application_identifier,
            image=screenshot_bytes,
            timestamp=int(time.time() * 1000),
            metadata={"perception_strategy": "native"},
        )

    async def get_current_application(self) -> str:
        """
        Resolve current iOS foreground application identifier.
        """

        return await self.__device.get_current_package()

    async def get_dimensions(self) -> Tuple[int, int]:
        """
        Return iOS screen dimensions.
        """

        return await self.__device.get_dimensions()


class IOSEnhancedPerceptionAdapter(PerceptionPort):
    """
    Enhanced iOS perception adapter with optional hierarchy extraction.
    """

    def __init__(
        self,
        *,
        device: IOSDevice,
        hierarchy: HierarchyPort,
    ) -> None:
        """
        Initialize enhanced iOS perception adapter.
        """

        self.__device = device
        self.__hierarchy = hierarchy

    @property
    def configuration(self) -> DeviceRuntimeConfiguration:
        """
        Return platform-neutral runtime configuration.
        """

        return self.__device.configuration

    async def capture(self) -> ScreenCapture:
        """
        Capture iOS screenshot with optional hierarchy enhancement.
        """

        results = await asyncio.gather(
            self.__device.capture_screen(),
            self.__hierarchy.dump_hierarchy(),
            return_exceptions=True,
        )

        screenshot_result = results[0]
        hierarchy_result = results[1]

        if isinstance(screenshot_result, Exception):
            raise DeviceError(f"iOS enhanced perception screenshot failed: {screenshot_result}")

        screenshot_bytes = cast("bytes", screenshot_result)
        hierarchy_content = hierarchy_result if isinstance(hierarchy_result, str) else None
        width, height = await self.__device.get_dimensions()
        application_identifier = (
            self.__resolve_application_identifier(hierarchy_content=hierarchy_content)
            or await self.__device.get_current_package()
        )

        metadata = {"perception_strategy": "enhanced"}
        if isinstance(hierarchy_result, Exception):
            metadata["hierarchy_error"] = str(hierarchy_result)

        return ScreenCapture(
            width=width,
            height=height,
            activity=application_identifier,
            image=screenshot_bytes,
            xml_content=hierarchy_content,
            timestamp=int(time.time() * 1000),
            metadata=metadata,
        )

    async def get_current_application(self) -> str:
        """
        Resolve current iOS foreground application identifier.
        """

        try:
            hierarchy_content = await self.__hierarchy.dump_hierarchy()
        except Exception:
            hierarchy_content = None

        return (
            self.__resolve_application_identifier(hierarchy_content=hierarchy_content)
            or await self.__device.get_current_package()
        )

    async def get_dimensions(self) -> Tuple[int, int]:
        """
        Return iOS screen dimensions.
        """

        return await self.__device.get_dimensions()

    def __resolve_application_identifier(self, *, hierarchy_content: str | None) -> str | None:
        """
        Extract the active iOS bundle identifier from the hierarchy XML when available.
        """

        if not hierarchy_content:
            return None

        try:
            root = ElementTree.fromstring(hierarchy_content)  # nosec
        except Exception:
            return None

        if root.tag == "XCUIElementTypeApplication":
            bundle_identifier = root.attrib.get("bundleId")
            if bundle_identifier:
                return str(bundle_identifier)

        application_node = root.find(".//*[@bundleId]")
        if application_node is None:
            return None

        bundle_identifier = application_node.attrib.get("bundleId")
        if not bundle_identifier:
            return None

        return str(bundle_identifier)
