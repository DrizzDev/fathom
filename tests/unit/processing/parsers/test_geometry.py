from __future__ import annotations

import unittest
import xml.etree.ElementTree as ET  # nosec

from fathom.processing.parsers.geometry import ElementGeometry, PixelRect


class PixelRectTest(unittest.TestCase):
    """
    Verifies rectangle area and point containment.
    """

    def test_area_is_width_times_height(self) -> None:
        rect = PixelRect(left=10, top=20, right=110, bottom=70)
        self.assertEqual(rect.area, 100 * 50)

    def test_inverted_extents_read_as_zero_area(self) -> None:
        rect = PixelRect(left=100, top=100, right=10, bottom=10)
        self.assertEqual(rect.area, 0)

    def test_contains_is_inclusive_of_edges(self) -> None:
        rect = PixelRect(left=0, top=0, right=100, bottom=100)
        self.assertTrue(rect.contains(x=0, y=0))
        self.assertTrue(rect.contains(x=50, y=50))
        self.assertTrue(rect.contains(x=100, y=100))
        self.assertFalse(rect.contains(x=101, y=50))


class ElementGeometryTest(unittest.TestCase):
    """
    Verifies rectangle extraction from Android bounds and iOS frame attributes.
    """

    def test_reads_android_bounds(self) -> None:
        element = ET.fromstring('<node bounds="[100,200][300,260]"/>')  # nosec
        rect = ElementGeometry.rect_of(element=element)
        self.assertEqual(rect, PixelRect(left=100, top=200, right=300, bottom=260))

    def test_reads_ios_frame_attributes(self) -> None:
        element = ET.fromstring('<node x="10" y="20" width="40" height="60"/>')  # nosec
        rect = ElementGeometry.rect_of(element=element)
        self.assertEqual(rect, PixelRect(left=10, top=20, right=50, bottom=80))

    def test_missing_geometry_is_a_zero_rect(self) -> None:
        element = ET.fromstring("<node/>")  # nosec
        rect = ElementGeometry.rect_of(element=element)
        self.assertIsNotNone(rect)
        assert rect is not None
        self.assertEqual(rect.area, 0)

    def test_non_integer_geometry_returns_none(self) -> None:
        element = ET.fromstring('<node width="wide" height="tall"/>')  # nosec
        self.assertIsNone(ElementGeometry.rect_of(element=element))


if __name__ == "__main__":
    unittest.main()
