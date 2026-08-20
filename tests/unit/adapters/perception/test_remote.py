from __future__ import annotations

import io
import unittest
from unittest.mock import AsyncMock, Mock

from PIL import Image

from fathom.adapters.perception.remote import RemotePerceptionAdapter
from fathom.core.exceptions import DeviceConnectionClosedError, DeviceError
from fathom.schemas.configuration import DeviceRuntimeConfiguration


class RemotePerceptionAdapterTest(unittest.IsolatedAsyncioTestCase):
    """
    Cover remote perception fallback behavior for snapshot capture.
    """

    @staticmethod
    def __screenshot(*, width: int, height: int) -> bytes:
        """
        Encode a minimal screenshot carrying the requested pixel dimensions.
        """

        buffer = io.BytesIO()
        Image.new("RGB", (width, height), "white").save(buffer, format="PNG")
        return buffer.getvalue()

    def __build_device(self) -> Mock:
        """
        Build a device double matching the remote perception contract.
        """

        device = Mock()

        device.configuration = DeviceRuntimeConfiguration()
        device.get_dimensions = AsyncMock(return_value=(1080, 2400))
        device.get_current_package = AsyncMock(return_value="in.delivery.android")

        return device

    async def test_capture_falls_back_to_separate_calls_when_snapshot_timeout_is_retryable(
        self,
    ) -> None:
        """
        Use screenshot and hierarchy fallback when atomic snapshot fails transiently.
        """

        device = self.__build_device()
        device.get_snapshot = AsyncMock(side_effect=DeviceError("timeout", retryable=True))
        device.capture_screen = AsyncMock(return_value=b"png-bytes")
        device.dump_hierarchy = AsyncMock(return_value="<hierarchy />")

        adapter = RemotePerceptionAdapter(device=device, include_hierarchy=True)

        capture = await adapter.capture()

        self.assertEqual(capture.image, b"png-bytes")
        self.assertEqual(capture.xml_content, "<hierarchy />")

        device.capture_screen.assert_awaited_once_with()
        device.dump_hierarchy.assert_awaited_once_with()

    async def test_capture_keeps_screenshot_when_hierarchy_fallback_fails(self) -> None:
        """
        Do not fail the whole capture when only hierarchy fallback fails.
        """

        device = self.__build_device()
        device.get_snapshot = AsyncMock(return_value=(b"", None))
        device.capture_screen = AsyncMock(return_value=b"png-bytes")
        device.dump_hierarchy = AsyncMock(
            side_effect=DeviceError("xml unavailable", retryable=True)
        )

        adapter = RemotePerceptionAdapter(device=device, include_hierarchy=True)

        capture = await adapter.capture()

        self.assertEqual(capture.image, b"png-bytes")
        self.assertIsNone(capture.xml_content)

    async def test_capture_raises_non_retryable_snapshot_failure(self) -> None:
        """
        Do not mask non-retryable snapshot failures such as closed clients.
        """

        device = self.__build_device()
        device.get_snapshot = AsyncMock(side_effect=DeviceConnectionClosedError("closed"))
        device.capture_screen = AsyncMock()
        device.dump_hierarchy = AsyncMock()

        adapter = RemotePerceptionAdapter(device=device, include_hierarchy=True)

        with self.assertRaises(DeviceError):
            await adapter.capture()

        device.capture_screen.assert_not_awaited()
        device.dump_hierarchy.assert_not_awaited()

    async def test_capture_corrects_logical_dims_for_landscape_image(self) -> None:
        """
        Landscape screenshot with portrait-reported dims must yield a landscape capture.
        """

        device = self.__build_device()
        device.get_dimensions = AsyncMock(return_value=(1080, 2340))
        device.get_snapshot = AsyncMock(
            return_value=(self.__screenshot(width=2340, height=1080), None),
        )

        adapter = RemotePerceptionAdapter(device=device, include_hierarchy=True)

        capture = await adapter.capture()

        self.assertEqual(capture.width, 2340)
        self.assertEqual(capture.height, 1080)
