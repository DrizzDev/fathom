from __future__ import annotations

import io
import unittest
from unittest.mock import patch

from PIL import Image

from fathom.adapters.device.local.ios import IOSDevice
from fathom.schemas.configuration import IOSConfiguration


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


@patch("fathom.adapters.device.local.ios.IOSAutomationGateway")
class IOSDeviceCacheDimensionsTest(unittest.TestCase):
    """
    Cover screenshot-PNG dimension caching introduced by the orientation fix.
    """

    def __build_device(self) -> IOSDevice:
        """
        Construct an IOSDevice with the automation gateway patched out.
        """

        return IOSDevice(configuration=IOSConfiguration())

    def test_caches_dimensions_from_valid_png(self, _gateway: object) -> None:
        """
        Populate the dimension cache from a parseable PNG header.
        """

        device = self.__build_device()

        device._IOSDevice__cache_dimensions_from_image(  # type: ignore[attr-defined]
            image=build_png(width=1179, height=2556),
        )

        cached = device._IOSDevice__cached_dimensions  # type: ignore[attr-defined]
        self.assertEqual(cached, (1179, 2556))

    def test_first_valid_png_wins_over_subsequent_pngs(self, _gateway: object) -> None:
        """
        Lock the cache to the first valid PNG dimensions and ignore later differing ones.
        """

        device = self.__build_device()

        device._IOSDevice__cache_dimensions_from_image(  # type: ignore[attr-defined]
            image=build_png(width=1179, height=2556),
        )
        device._IOSDevice__cache_dimensions_from_image(  # type: ignore[attr-defined]
            image=build_png(width=393, height=852),
        )

        cached = device._IOSDevice__cached_dimensions  # type: ignore[attr-defined]
        self.assertEqual(cached, (1179, 2556))

    def test_invalid_png_leaves_cache_unset(self, _gateway: object) -> None:
        """
        Swallow PNG parse errors silently and leave the cache untouched.
        """

        device = self.__build_device()

        device._IOSDevice__cache_dimensions_from_image(  # type: ignore[attr-defined]
            image=b"not-a-valid-png",
        )

        cached = device._IOSDevice__cached_dimensions  # type: ignore[attr-defined]
        self.assertIsNone(cached)

    def test_zero_dimension_png_leaves_cache_unset(self, _gateway: object) -> None:
        """
        Reject zero-dimension PNG headers without setting the cache.
        """

        device = self.__build_device()

        device._IOSDevice__cache_dimensions_from_image(  # type: ignore[attr-defined]
            image=build_png(width=0, height=0),
        )

        cached = device._IOSDevice__cached_dimensions  # type: ignore[attr-defined]
        self.assertIsNone(cached)

    def test_jpeg_falls_back_to_pil_and_caches_dimensions(self, _gateway: object) -> None:
        """
        Non-PNG bytes must route through the PIL fallback (added in
        7eb22e5) and still cache the decoded dimensions.
        """

        jpeg_buffer = io.BytesIO()
        Image.new("RGB", (1170, 2532)).save(jpeg_buffer, format="JPEG", quality=1)

        device = self.__build_device()

        device._IOSDevice__cache_dimensions_from_image(  # type: ignore[attr-defined]
            image=jpeg_buffer.getvalue(),
        )

        cached = device._IOSDevice__cached_dimensions  # type: ignore[attr-defined]
        self.assertEqual(cached, (1170, 2532))


@patch("fathom.adapters.device.local.ios.IOSAutomationGateway")
class IOSDeviceIDBDispatchTest(unittest.IsolatedAsyncioTestCase):
    """
    Verify that the IDB backend routes interactions through ``IDBClient``
    and that the device's resolved UDID is forwarded to the client.
    """

    async def test_tap_routes_to_idb_client_with_pixel_to_point_conversion(
        self, _gateway: object
    ) -> None:
        """
        idb's ``ui tap`` operates in UIKit points, not screenshot pixels;
        ``IOSDevice.tap`` must scale incoming pixel coords by the
        ``width_points``/``height_points`` ratio reported by idb
        describe.
        """

        from unittest.mock import AsyncMock as AsyncMockLocal

        from fathom.constants.platform import IOSAutomationBackend
        from fathom.schemas.configuration import IOSConfiguration

        device = IOSDevice(
            configuration=IOSConfiguration(
                device_identifier="UDID-1",
                automation_backend=IOSAutomationBackend.IDB,
            )
        )

        idb_stub = device._IOSDevice__idb_client  # type: ignore[attr-defined]
        assert idb_stub is not None
        # PNG dims 1206×2622 paired with point dims 402×874 (iPhone 17
        # at density 3.0) → each pixel input must be divided by 3 on the
        # way to idb.
        idb_stub.capture_screen = AsyncMockLocal(
            return_value=build_png(width=1206, height=2622) + b"\x00" * 1024,
        )
        idb_stub.describe = AsyncMockLocal(return_value=(402, 874))
        idb_stub.tap = AsyncMockLocal(return_value=None)

        result = await device.tap(x=126, y=252)

        self.assertTrue(result.success)
        idb_stub.tap.assert_awaited_once_with(x=42.0, y=84.0)

    async def test_swipe_routes_to_idb_client_with_pixel_to_point_conversion(
        self, _gateway: object
    ) -> None:
        """
        Both swipe endpoints must be converted from screenshot pixels
        to idb HID points before reaching ``IDBClient.swipe``.
        """

        from unittest.mock import AsyncMock as AsyncMockLocal

        from fathom.constants.platform import IOSAutomationBackend
        from fathom.schemas.configuration import IOSConfiguration

        device = IOSDevice(
            configuration=IOSConfiguration(
                device_identifier="UDID-1",
                automation_backend=IOSAutomationBackend.IDB,
            )
        )

        idb_stub = device._IOSDevice__idb_client  # type: ignore[attr-defined]
        assert idb_stub is not None
        idb_stub.capture_screen = AsyncMockLocal(
            return_value=build_png(width=1206, height=2622) + b"\x00" * 1024,
        )
        idb_stub.describe = AsyncMockLocal(return_value=(402, 874))
        idb_stub.swipe = AsyncMockLocal(return_value=None)

        result = await device.swipe(x1=126, y1=252, x2=300, y2=600, duration=500)

        self.assertTrue(result.success)
        idb_stub.swipe.assert_awaited_once_with(
            x1=42.0,
            y1=84.0,
            x2=100.0,
            y2=200.0,
            duration_milliseconds=500,
        )

    async def test_capture_screen_routes_to_idb(self, _gateway: object) -> None:
        from unittest.mock import AsyncMock as AsyncMockLocal

        from fathom.constants.platform import IOSAutomationBackend
        from fathom.schemas.configuration import IOSConfiguration

        device = IOSDevice(
            configuration=IOSConfiguration(
                device_identifier="UDID-1",
                automation_backend=IOSAutomationBackend.IDB,
            )
        )
        idb_stub = device._IOSDevice__idb_client  # type: ignore[attr-defined]
        png = build_png(width=1170, height=2532) + b"\x00" * 1024
        idb_stub.capture_screen = AsyncMockLocal(return_value=png)

        result = await device.capture_screen()

        self.assertEqual(result, png)
        idb_stub.capture_screen.assert_awaited_once()

    async def test_dump_hierarchy_raises_on_idb_backend(self, _gateway: object) -> None:
        from fathom.constants.platform import IOSAutomationBackend
        from fathom.core.exceptions import DeviceError
        from fathom.schemas.configuration import IOSConfiguration

        device = IOSDevice(
            configuration=IOSConfiguration(
                device_identifier="UDID-1",
                automation_backend=IOSAutomationBackend.IDB,
            )
        )
        with self.assertRaises(DeviceError) as ctx:
            await device.dump_hierarchy()
        self.assertIn("XCUIElement", str(ctx.exception))

    async def test_launch_configured_package_launches_bundle_on_idb(self, _gateway: object) -> None:
        from unittest.mock import AsyncMock as AsyncMockLocal

        from fathom.constants.platform import IOSAutomationBackend
        from fathom.schemas.configuration import IOSConfiguration

        device = IOSDevice(
            configuration=IOSConfiguration(
                device_identifier="UDID-1",
                automation_backend=IOSAutomationBackend.IDB,
                bundle_identifier="com.example.app",
            )
        )
        idb_stub = device._IOSDevice__idb_client  # type: ignore[attr-defined]
        idb_stub.launch = AsyncMockLocal(return_value=None)

        await device.launch_configured_package()

        idb_stub.launch.assert_awaited_once_with(bundle_identifier="com.example.app")

    async def test_launch_configured_package_skips_without_bundle(self, _gateway: object) -> None:
        from unittest.mock import AsyncMock as AsyncMockLocal

        from fathom.constants.platform import IOSAutomationBackend
        from fathom.schemas.configuration import IOSConfiguration

        device = IOSDevice(
            configuration=IOSConfiguration(
                device_identifier="UDID-1",
                automation_backend=IOSAutomationBackend.IDB,
                bundle_identifier=None,
            )
        )
        idb_stub = device._IOSDevice__idb_client  # type: ignore[attr-defined]
        idb_stub.launch = AsyncMockLocal(return_value=None)

        await device.launch_configured_package()

        idb_stub.launch.assert_not_awaited()

    async def test_launch_configured_package_no_op_on_non_idb_backend(
        self, _gateway: object
    ) -> None:
        from fathom.constants.platform import IOSAutomationBackend
        from fathom.schemas.configuration import IOSConfiguration

        device = IOSDevice(
            configuration=IOSConfiguration(
                device_identifier="UDID-1",
                automation_backend=IOSAutomationBackend.XCRUN_SIMCTL,
                bundle_identifier="com.example.app",
            )
        )

        # Should be a silent no-op — no idb client to invoke.
        await device.launch_configured_package()

    async def test_terminate_configured_package_invokes_idb_terminate(
        self, _gateway: object
    ) -> None:
        from unittest.mock import AsyncMock as AsyncMockLocal

        from fathom.constants.platform import IOSAutomationBackend
        from fathom.schemas.configuration import IOSConfiguration

        device = IOSDevice(
            configuration=IOSConfiguration(
                device_identifier="UDID-1",
                automation_backend=IOSAutomationBackend.IDB,
                bundle_identifier="com.example.app",
            )
        )
        idb_stub = device._IOSDevice__idb_client  # type: ignore[attr-defined]
        idb_stub.terminate = AsyncMockLocal(return_value=None)

        await device.terminate_configured_package()

        idb_stub.terminate.assert_awaited_once_with(bundle_identifier="com.example.app")

    async def test_terminate_configured_package_no_op_on_non_idb_backend(
        self, _gateway: object
    ) -> None:
        from fathom.constants.platform import IOSAutomationBackend
        from fathom.schemas.configuration import IOSConfiguration

        device = IOSDevice(
            configuration=IOSConfiguration(
                device_identifier="UDID-1",
                automation_backend=IOSAutomationBackend.XCRUN_SIMCTL,
                bundle_identifier="com.example.app",
            )
        )

        # Silent no-op; no idb client exists on this backend.
        await device.terminate_configured_package()

    async def test_terminate_swallows_device_error(self, _gateway: object) -> None:
        from unittest.mock import AsyncMock as AsyncMockLocal

        from fathom.constants.platform import IOSAutomationBackend
        from fathom.core.exceptions import DeviceError
        from fathom.schemas.configuration import IOSConfiguration

        device = IOSDevice(
            configuration=IOSConfiguration(
                device_identifier="UDID-1",
                automation_backend=IOSAutomationBackend.IDB,
                bundle_identifier="com.example.app",
            )
        )
        idb_stub = device._IOSDevice__idb_client  # type: ignore[attr-defined]
        idb_stub.terminate = AsyncMockLocal(side_effect=DeviceError("not running"))

        # Must not raise — finally-block cleanup can't afford to throw.
        await device.terminate_configured_package()

    async def test_launch_configured_package_is_idempotent(self, _gateway: object) -> None:
        from unittest.mock import AsyncMock as AsyncMockLocal

        from fathom.constants.platform import IOSAutomationBackend
        from fathom.schemas.configuration import IOSConfiguration

        device = IOSDevice(
            configuration=IOSConfiguration(
                device_identifier="UDID-1",
                automation_backend=IOSAutomationBackend.IDB,
                bundle_identifier="com.example.app",
            )
        )
        idb_stub = device._IOSDevice__idb_client  # type: ignore[attr-defined]
        idb_stub.launch = AsyncMockLocal(return_value=None)

        await device.launch_configured_package()
        await device.launch_configured_package()
        await device.launch_configured_package()

        # Guarded: launch fires once even if called repeatedly.
        idb_stub.launch.assert_awaited_once()


@patch("fathom.adapters.device.local.ios.IOSAutomationGateway")
class IOSDeviceGetDimensionsTest(unittest.IsolatedAsyncioTestCase):
    """
    Cover get_dimensions cached fast-path, avoiding any simctl invocation.
    """

    async def test_returns_cached_dimensions_without_capturing(self, _gateway: object) -> None:
        """
        Skip screenshot capture entirely when dimensions are already cached.
        """

        device = IOSDevice(configuration=IOSConfiguration())
        device._IOSDevice__cached_dimensions = (1179, 2556)  # type: ignore[attr-defined]

        result = await device.get_dimensions()

        self.assertEqual(result, (1179, 2556))
