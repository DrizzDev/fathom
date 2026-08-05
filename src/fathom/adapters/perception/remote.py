from __future__ import annotations

import time
from logging import getLogger
from typing import Dict, Optional, Tuple

from fathom.adapters.perception.breaker import HierarchyBreaker
from fathom.constants.screen import HierarchyProvenance
from fathom.core.exceptions import DeviceError
from fathom.core.perception.orientation import CaptureOrientationResolver
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

        self.__breaker = HierarchyBreaker()

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
        provenance: Optional[HierarchyProvenance] = None

        if self.__include_hierarchy:
            snapshot = await self.__breaker.snapshot(
                dump=self.__capture,
                screenshot=self.__device.capture_screen,
            )

            provenance = snapshot.provenance
            screenshot_bytes = snapshot.image
            hierarchy_content = snapshot.hierarchy
        else:
            hierarchy_content = None
            screenshot_bytes = await self.__device.capture_screen()

        if not screenshot_bytes:
            raise DeviceError("Remote perception captured an empty screenshot.")

        reported_width, reported_height = await self.__device.get_dimensions()
        width, height = CaptureOrientationResolver.resolve(
            image=screenshot_bytes,
            reported_width=reported_width,
            reported_height=reported_height,
        )
        application_identifier = await self.__device.get_current_package()

        elapsed = time.time() - capture_start
        metadata: Dict[str, object] = {"capture_duration": elapsed}

        if hierarchy_content is not None:
            metadata["hierarchy_dump_duration"] = elapsed

        if provenance is not None:
            metadata["hierarchy_fallback"] = provenance.value

        return ScreenCapture(
            width=width,
            height=height,
            metadata=metadata,
            image=screenshot_bytes,
            xml_content=hierarchy_content,
            activity=application_identifier,
            timestamp=int(time.time() * 1000),
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
