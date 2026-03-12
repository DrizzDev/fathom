from __future__ import annotations

import unittest
from typing import List, Optional, Tuple, cast
from unittest.mock import patch

from fathom.adapters.device.remote.adb import ADBRemoteDeviceAdapter
from fathom.adapters.device.remote.ios import IOSRemoteDeviceAdapter
from fathom.constants.interaction import SwipeSpeed
from fathom.constants.platform import DeviceConnectionType, DevicePlatform
from fathom.runtime.factories import DeviceFactory
from fathom.schemas.configuration import (
    DeviceConfiguration,
    DeviceRuntimeConfiguration,
    RemoteDeviceConfiguration,
)
from fathom.schemas.results import ActionResult


def build_png(*, width: int, height: int) -> bytes:
    """Build a minimal PNG header containing the requested dimensions."""

    return (
        b"\x89PNG\r\n\x1a\n"
        + b"\x00\x00\x00\r"
        + b"IHDR"
        + width.to_bytes(4, byteorder="big")
        + height.to_bytes(4, byteorder="big")
        + b"\x08\x02\x00\x00\x00"
    )


class FakeADBRemoteDeviceAdapter:
    """Test double that mimics the remote Android transport adapter contract."""

    def __init__(
        self,
        *,
        screenshot_dimensions: Tuple[int, int],
        automation_dimensions: Tuple[int, int],
    ) -> None:
        """Initialize fake remote adapter state."""

        self.configuration = DeviceRuntimeConfiguration(platform=DevicePlatform.IOS)
        self.tap_calls: List[Tuple[int, int]] = []
        self.swipe_calls: List[Tuple[int, int, int, int, Optional[int], Optional[SwipeSpeed]]] = []
        self.screenshot = build_png(
            width=screenshot_dimensions[0],
            height=screenshot_dimensions[1],
        )
        self.automation_dimensions = automation_dimensions

    async def tap(self, *, x: int, y: int) -> ActionResult:
        """Record tap coordinates."""

        self.tap_calls.append((x, y))
        return ActionResult(success=True, duration=1)

    async def type(self, *, text: str) -> ActionResult:
        """Ignore text input for tests."""

        return ActionResult(success=True, duration=1)

    async def swipe(
        self,
        *,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        duration: Optional[int] = None,
        speed: Optional[SwipeSpeed] = None,
    ) -> ActionResult:
        """Record swipe coordinates."""

        self.swipe_calls.append((x1, y1, x2, y2, duration, speed))
        return ActionResult(success=True, duration=1)

    async def back(self) -> ActionResult:
        """Return success for unsupported test action."""

        return ActionResult(success=True, duration=1)

    async def home(self) -> ActionResult:
        """Return success for unsupported test action."""

        return ActionResult(success=True, duration=1)

    async def get_current_package(self) -> str:
        """Return a stable foreground package."""

        return "com.apple.springboard"

    async def capture_screen(self) -> bytes:
        """Return cached screenshot bytes."""

        return self.screenshot

    async def dump_hierarchy(self) -> Optional[str]:
        """Return no hierarchy for test."""

        return None

    async def get_snapshot(self) -> Tuple[bytes, Optional[str]]:
        """Return screenshot-only snapshot."""

        return self.screenshot, None

    async def get_dimensions(self) -> Tuple[int, int]:
        """Return logical automation dimensions."""

        return self.automation_dimensions

    async def wait_for_device(self, *, timeout: float) -> bool:
        """Always report ready."""

        return True

    async def close(self) -> None:
        """No-op close."""

        return None


class IOSRemoteDeviceAdapterTest(unittest.IsolatedAsyncioTestCase):
    """Cover iOS-specific remote coordinate conversion."""

    async def test_tap_converts_screenshot_pixels_to_automation_points(self) -> None:
        """Convert screenshot-space tap coordinates before delegating transport."""

        delegate = FakeADBRemoteDeviceAdapter(
            screenshot_dimensions=(1179, 2556),
            automation_dimensions=(393, 852),
        )
        adapter = IOSRemoteDeviceAdapter(
            configuration=DeviceConfiguration(
                type=DeviceConnectionType.REMOTE,
                platform=DevicePlatform.IOS,
            ),
            delegate=cast("ADBRemoteDeviceAdapter", delegate),
        )

        await adapter.tap(x=999, y=666)

        self.assertEqual(delegate.tap_calls, [(333, 222)])


class DeviceFactoryTest(unittest.TestCase):
    """Keep Android and iOS remote device selection distinct."""

    def test_remote_ios_uses_ios_remote_adapter(self) -> None:
        """Select the iOS-specific remote adapter for remote iOS runs."""

        factory = DeviceFactory()
        with patch("fathom.adapters.device.remote.adb.httpx.AsyncClient"):
            device = factory.create(
                configuration=DeviceConfiguration(
                    type=DeviceConnectionType.REMOTE,
                    platform=DevicePlatform.IOS,
                    remote=RemoteDeviceConfiguration(
                        provider_url="https://example.test",
                        session_id="session-id",
                    ),
                )
            )

        self.assertIsInstance(device, IOSRemoteDeviceAdapter)

    def test_remote_android_uses_android_remote_adapter(self) -> None:
        """Keep Android remote runs on the transport-only adapter."""

        factory = DeviceFactory()
        with patch("fathom.adapters.device.remote.adb.httpx.AsyncClient"):
            device = factory.create(
                configuration=DeviceConfiguration(
                    type=DeviceConnectionType.REMOTE,
                    platform=DevicePlatform.ANDROID,
                    remote=RemoteDeviceConfiguration(
                        provider_url="https://example.test",
                        session_id="session-id",
                    ),
                )
            )

        self.assertIsInstance(device, ADBRemoteDeviceAdapter)
