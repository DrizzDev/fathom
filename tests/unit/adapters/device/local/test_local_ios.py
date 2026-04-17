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
