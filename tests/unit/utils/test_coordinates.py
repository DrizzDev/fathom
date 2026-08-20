from __future__ import annotations

import unittest

from fathom.schemas.actions import Bounds, CoordinateSource, CoordinateSystem
from fathom.utils.coordinates import CoordinateConverter


class CoordinateConverterTest(unittest.TestCase):
    """
    Unit tests for region-based gesture path derivation.
    """

    def test_model_region_works_without_xml(self) -> None:
        """
        Derive a swipe path from a model bbox when XML is unavailable.
        """

        converter = CoordinateConverter(logical_width=1080, logical_height=2340)
        # Magnitudes already in screen-pixel space, so the producer must declare
        # DEVICE_PIXEL rather than rely on any normalized-vs-pixel inference.
        bounds = Bounds(
            x=44,
            y=1983,
            width=250,
            height=116,
            coordinate_system=CoordinateSystem.DEVICE_PIXEL,
        )

        region = converter.region_from_bounds(bounds=bounds, source=CoordinateSource.MODEL)
        path = converter.resolve_swipe_path(region=region, direction="right")

        self.assertEqual(region.source, "model")
        self.assertEqual((region.x, region.y, region.width, region.height), (44, 1983, 250, 116))

        self.assertEqual(path.distance, 210)
        self.assertEqual(path.to_coordinates(), (64, 2041, 274, 2041))

    def test_capture_bounds_scale_logical_region_to_retina_pixels(self) -> None:
        """
        Translate a logical viewport region into capture-space pixel bounds.
        """

        converter = CoordinateConverter(
            logical_width=430,
            logical_height=932,
            pixel_width=1290,
            pixel_height=2796,
        )

        region = converter.viewport_region()
        bounds = converter.capture_bounds(region=region)

        self.assertEqual(bounds.system, CoordinateSystem.DEVICE_PIXEL)
        self.assertEqual((bounds.x, bounds.y, bounds.width, bounds.height), (0, 0, 1290, 2796))


if __name__ == "__main__":
    unittest.main()
