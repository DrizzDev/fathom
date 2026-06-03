from __future__ import annotations

import io
import unittest
from unittest.mock import AsyncMock, Mock

from PIL import Image

from fathom.adapters.perception.android import AndroidPerceptionAdapter
from fathom.schemas.configuration import DeviceRuntimeConfiguration


class AndroidPerceptionAdapterTest(unittest.IsolatedAsyncioTestCase):
    """
    Cover the orientation-correcting capture path for Android perception.
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
        Build a device double whose contract matches the Android perception adapter.
        """

        device = Mock()
        device.configuration = DeviceRuntimeConfiguration()
        device.get_dimensions = AsyncMock(return_value=reported)
        device.get_current_package = AsyncMock(return_value="com.example.app")
        return device

    async def test_landscape_screenshot_swaps_portrait_report(self) -> None:
        """
        Cooking-Craze-style mismatch: device cache says portrait, screenshot is landscape.
        """

        screenshot = self.__screenshot(width=2340, height=1080)
        device = self.__device(reported=(1080, 2340))
        device.capture_screen = AsyncMock(return_value=screenshot)
        device.get_snapshot = AsyncMock(return_value=(screenshot, None))

        adapter = AndroidPerceptionAdapter(device=device, include_hierarchy=False)

        capture = await adapter.capture()

        self.assertEqual(capture.width, 2340)
        self.assertEqual(capture.height, 1080)

    async def test_aligned_orientations_preserve_reported_dims(self) -> None:
        """
        When the report already matches the screenshot aspect, no swap occurs.
        """

        screenshot = self.__screenshot(width=1080, height=2340)
        device = self.__device(reported=(1080, 2340))
        device.capture_screen = AsyncMock(return_value=screenshot)
        device.get_snapshot = AsyncMock(return_value=(screenshot, None))

        adapter = AndroidPerceptionAdapter(device=device, include_hierarchy=False)

        capture = await adapter.capture()

        self.assertEqual(capture.width, 1080)
        self.assertEqual(capture.height, 2340)
