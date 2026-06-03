from __future__ import annotations

import io
import unittest
from unittest.mock import AsyncMock, Mock

from PIL import Image

from fathom.adapters.perception.ios import (
    IOSEnhancedPerceptionAdapter,
    IOSNativePerceptionAdapter,
)
from fathom.schemas.configuration import DeviceRuntimeConfiguration


class IOSNativePerceptionAdapterTest(unittest.IsolatedAsyncioTestCase):
    """
    Cover the orientation-correcting capture path for iOS native perception.
    """

    @staticmethod
    def __screenshot(*, width: int, height: int) -> bytes:
        """
        Encode a minimal screenshot carrying the requested pixel dimensions.
        """

        buffer = io.BytesIO()
        Image.new("RGB", (width, height), "white").save(buffer, format="PNG")
        return buffer.getvalue()

    @classmethod
    def __device(cls, *, reported: tuple[int, int]) -> Mock:
        """
        Build a device double whose contract matches the iOS perception adapters.
        """

        device = Mock()
        device.configuration = DeviceRuntimeConfiguration()
        device.get_dimensions = AsyncMock(return_value=reported)
        device.get_current_package = AsyncMock(return_value="com.apple.example")
        return device

    async def test_retina_landscape_screenshot_swaps_portrait_logical(self) -> None:
        """
        iOS retina landscape pixels with portrait-reported points must yield landscape points.
        """

        screenshot = self.__screenshot(width=2436, height=1125)
        device = self.__device(reported=(375, 812))
        device.capture_screen = AsyncMock(return_value=screenshot)

        adapter = IOSNativePerceptionAdapter(device=device)

        capture = await adapter.capture()

        self.assertEqual(capture.width, 812)
        self.assertEqual(capture.height, 375)

    async def test_aligned_portrait_orientations_preserve_reported_dims(self) -> None:
        """
        Retina portrait pixels with portrait logical points pass through unchanged.
        """

        screenshot = self.__screenshot(width=1125, height=2436)
        device = self.__device(reported=(375, 812))
        device.capture_screen = AsyncMock(return_value=screenshot)

        adapter = IOSNativePerceptionAdapter(device=device)

        capture = await adapter.capture()

        self.assertEqual(capture.width, 375)
        self.assertEqual(capture.height, 812)


class IOSEnhancedPerceptionAdapterTest(unittest.IsolatedAsyncioTestCase):
    """
    Cover the orientation-correcting capture path for iOS enhanced perception.
    """

    @staticmethod
    def __screenshot(*, width: int, height: int) -> bytes:
        """
        Encode a minimal screenshot carrying the requested pixel dimensions.
        """

        buffer = io.BytesIO()
        Image.new("RGB", (width, height), "white").save(buffer, format="PNG")
        return buffer.getvalue()

    @classmethod
    def __device(cls, *, reported: tuple[int, int]) -> Mock:
        """
        Build a device double whose contract matches the iOS enhanced adapter.
        """

        device = Mock()
        device.configuration = DeviceRuntimeConfiguration()
        device.get_dimensions = AsyncMock(return_value=reported)
        device.get_current_package = AsyncMock(return_value="com.apple.example")
        return device

    async def test_retina_landscape_screenshot_swaps_portrait_logical(self) -> None:
        """
        Enhanced snapshot with landscape pixels and portrait points lands as landscape points.
        """

        screenshot = self.__screenshot(width=2436, height=1125)
        device = self.__device(reported=(375, 812))
        device.get_snapshot = AsyncMock(return_value=(screenshot, "<hierarchy />"))

        adapter = IOSEnhancedPerceptionAdapter(device=device)

        capture = await adapter.capture()

        self.assertEqual(capture.width, 812)
        self.assertEqual(capture.height, 375)
        self.assertEqual(capture.xml_content, "<hierarchy />")
