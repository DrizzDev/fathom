from __future__ import annotations

import time
from typing import Optional

from fathom.adapters.perception.breaker import HierarchyBreaker
from fathom.constants.screen import HierarchyProvenance
from fathom.core.exceptions import DeviceError
from fathom.core.perception.orientation import CaptureOrientationResolver
from fathom.interfaces.device import DevicePort
from fathom.interfaces.perception import PerceptionPort
from fathom.schemas.configuration import DeviceRuntimeConfiguration
from fathom.schemas.observation import KeyboardObservation
from fathom.schemas.screens import ScreenCapture


class AndroidPerceptionAdapter(PerceptionPort):
    """
    Android perception adapter backed by the existing ADB device implementation.
    """

    def __init__(self, *, device: DevicePort, include_hierarchy: bool) -> None:
        """
        Bind the device adapter, whether to include the hierarchy dump, and a fresh hierarchy breaker.
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
        Capture Android screenshot and optional hierarchy in a single snapshot.
        """

        capture_start = time.time()
        provenance: Optional[HierarchyProvenance] = None

        if self.__include_hierarchy:
            snapshot = await self.__breaker.snapshot(
                dump=self.__device.get_snapshot,
                screenshot=self.__device.capture_screen,
            )
            provenance = snapshot.provenance
            screenshot_bytes = snapshot.image
            hierarchy_content = snapshot.hierarchy
        else:
            hierarchy_content = None
            screenshot_bytes = await self.__device.capture_screen()

        if not screenshot_bytes:
            raise DeviceError("Android perception captured an empty screenshot.")

        reported_width, reported_height = await self.__device.get_dimensions()
        width, height = CaptureOrientationResolver.resolve(
            image=screenshot_bytes,
            reported_width=reported_width,
            reported_height=reported_height,
        )

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
            metadata=self.__metadata(
                provenance=provenance,
                capture_start=capture_start,
                hierarchy_content=hierarchy_content,
            ),
        )

    @staticmethod
    def __metadata(
        *,
        capture_start: float,
        hierarchy_content: Optional[str],
        provenance: Optional[HierarchyProvenance],
    ) -> dict[str, object]:
        """
        Build capture metadata, recording hierarchy timing and any breaker fallback provenance.
        """

        elapsed = time.time() - capture_start
        metadata: dict[str, object] = {"capture_duration": elapsed}

        if hierarchy_content is not None:
            metadata["hierarchy_dump_duration"] = elapsed

        if provenance is not None:
            metadata["hierarchy_fallback"] = provenance.value

        return metadata

    async def detect_keyboard(
        self, *, capture: Optional[ScreenCapture] = None
    ) -> KeyboardObservation:
        """
        Delegate keyboard detection to the underlying device adapter (dumpsys for local ADB).
        """

        _ = capture
        return await self.__device.detect_keyboard()
