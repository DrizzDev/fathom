from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, Mock

from fathom.adapters.perception.remote import RemotePerceptionAdapter
from fathom.core.exceptions import DeviceConnectionClosedError, DeviceError
from fathom.schemas.configuration import DeviceRuntimeConfiguration


def build_png(*, width: int, height: int) -> bytes:
    """
    Build a minimal PNG header containing the requested dimensions.
    """

    return (
        b"\x89PNG\r\n\x1a\n"
        + b"\x00\x00\x00\r"
        + b"IHDR"
        + width.to_bytes(4, byteorder="big")
        + height.to_bytes(4, byteorder="big")
        + b"\x08\x02\x00\x00\x00"
    )


class RemotePerceptionAdapterTest(unittest.IsolatedAsyncioTestCase):
    """
    Cover remote perception fallback behavior for snapshot capture.
    """

    def __build_device(self) -> Mock:
        """
        Build a device double matching the remote perception contract.
        """

        device = Mock()

        device.configuration = DeviceRuntimeConfiguration()
        device.get_dimensions = AsyncMock(return_value=(1080, 2400))
        device.get_current_package = AsyncMock(return_value="in.swiggy.android")

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

    async def test_capture_uses_png_dimensions_over_device_get_dimensions(self) -> None:
        """
        Trust the screenshot PNG IHDR over the device's reported dimensions.
        """

        device = self.__build_device()
        device.get_snapshot = AsyncMock(
            return_value=(build_png(width=1179, height=2556), None),
        )

        adapter = RemotePerceptionAdapter(device=device, include_hierarchy=True)

        capture = await adapter.capture()

        self.assertEqual((capture.width, capture.height), (1179, 2556))
        device.get_dimensions.assert_not_awaited()

    async def test_capture_falls_back_to_device_dimensions_on_invalid_png(self) -> None:
        """
        Fall back to device.get_dimensions only when PNG header parse fails.
        """

        device = self.__build_device()
        device.get_snapshot = AsyncMock(return_value=(b"not-a-valid-png", None))

        adapter = RemotePerceptionAdapter(device=device, include_hierarchy=True)

        capture = await adapter.capture()

        self.assertEqual((capture.width, capture.height), (1080, 2400))
        device.get_dimensions.assert_awaited_once_with()
