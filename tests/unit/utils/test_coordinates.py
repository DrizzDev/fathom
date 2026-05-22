from __future__ import annotations

import re
import unittest
import xml.etree.ElementTree as ElementTree
from pathlib import Path

from fathom.schemas.actions import Bounds, CoordinateSource, CoordinateSystem
from fathom.utils.coordinates import CoordinateConverter

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class CoordinateConverterTest(unittest.TestCase):
    """
    Unit tests for region-based gesture path derivation.
    """

    def test_rolodex_knob_xml_region_uses_edge_based_swipe_path(self) -> None:
        """
        Derive the swipe path from the real Rolodex knob XML bounds.
        """

        bounds = self.__rolodex_knob_bounds()
        converter = CoordinateConverter(logical_width=1080, logical_height=2340)

        region = converter.region_from_bounds(bounds=bounds, source=CoordinateSource.XML)
        path = converter.resolve_swipe_path(region=region, direction="right")

        self.assertEqual(region.source, "xml")
        self.assertEqual((region.x, region.y, region.width, region.height), (66, 2010, 132, 62))

        self.assertEqual(path.distance, 100)
        self.assertEqual(path.to_coordinates(), (82, 2041, 182, 2041))

    def test_save_to_draft_xml_region_uses_full_region_width(self) -> None:
        """
        Derive a wide path from the real Save to draft XML bounds.
        """

        bounds = self.__save_to_draft_bounds()
        converter = CoordinateConverter(logical_width=1080, logical_height=2340)

        region = converter.region_from_bounds(bounds=bounds, source=CoordinateSource.XML)
        path = converter.resolve_swipe_path(region=region, direction="right")

        self.assertEqual(region.source, "xml")
        self.assertEqual((region.x, region.y, region.width, region.height), (44, 1817, 992, 143))

        self.assertEqual(path.distance, 864)
        self.assertEqual(path.to_coordinates(), (108, 1888, 972, 1888))

    def test_viewport_scroll_uses_real_screenshot_dimensions(self) -> None:
        """
        Derive scroll distance from the real screenshot viewport.
        """

        width, height = self.__screen_dimensions(
            PROJECT_ROOT / "tests/fixtures/coordinate/android/swipe/001/dimensions.txt"
        )
        converter = CoordinateConverter(logical_width=width, logical_height=height)

        region = converter.viewport_region()
        path = converter.resolve_scroll_path(region=region, direction="down")

        self.assertEqual(region.source, "viewport")
        self.assertEqual((region.x, region.y, region.width, region.height), (0, 0, 1080, 2340))

        self.assertEqual(path.distance, 2020)
        self.assertEqual(path.to_coordinates(), (540, 2180, 540, 160))

    def test_model_region_works_without_xml(self) -> None:
        """
        Derive a swipe path from a real Gemini bbox when XML is unavailable.
        """

        converter = CoordinateConverter(logical_width=1080, logical_height=2340)
        # The original fixture stamped these device-pixel values as
        # "normalized" because the old is_normalized heuristic silently
        # re-classified them. With explicit coordinate systems the
        # producer must declare DEVICE_PIXEL when the magnitudes are
        # already in screen-pixel space.
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

    @staticmethod
    def __screen_dimensions(path: Path) -> tuple[int, int]:
        """
        Read captured screenshot dimensions from a stable fixture.
        """

        value = path.read_text(encoding="utf-8").strip()
        match = re.fullmatch(r"(\d+)x(\d+)", value)

        if not match:
            raise AssertionError(f"Invalid dimensions fixture: {path}")

        width, height = (int(group) for group in match.groups())
        return width, height

    @staticmethod
    def __rolodex_knob_bounds() -> Bounds:
        """
        Return the real Rolodex knob bounds from coordinate fixture 001.
        """

        path = PROJECT_ROOT / "tests/fixtures/coordinate/android/swipe/001/hierarchy.xml"
        root = ElementTree.parse(path).getroot()

        for node in root.iter("node"):
            children = list(node)
            has_rolodex_text = any(
                child.attrib.get("text") == "Add to rolodex" for child in children
            )
            if not has_rolodex_text:
                continue

            for child in children:
                if (
                    child.attrib.get("text") == ""
                    and child.attrib.get("class") == "android.view.View"
                    and child.attrib.get("bounds") == "[66,2010][198,2072]"
                ):
                    return CoordinateConverterTest.__bounds_from_xml(child.attrib["bounds"])

        raise AssertionError(f"Could not find Rolodex knob bounds in {path}")

    @staticmethod
    def __save_to_draft_bounds() -> Bounds:
        """
        Return the real Save to draft button bounds from coordinate fixture 002.
        """

        path = PROJECT_ROOT / "tests/fixtures/coordinate/android/swipe/002/hierarchy.xml"
        root = ElementTree.parse(path).getroot()

        for node in root.iter("node"):
            children = list(node)
            has_save_text = any(
                child.attrib.get("text") == "Save to draft (to complete later)"
                for child in children
            )
            has_button_shape = any(
                child.attrib.get("class") == "android.widget.Button"
                and child.attrib.get("bounds") == "[44,1817][1036,1960]"
                for child in children
            )

            if has_save_text and has_button_shape:
                return CoordinateConverterTest.__bounds_from_xml(node.attrib["bounds"])

        raise AssertionError(f"Could not find Save to draft text in {path}")

    @staticmethod
    def __bounds_from_xml(value: str) -> Bounds:
        """
        Convert an Android XML bounds string into pixel bounds.
        """

        match = re.fullmatch(r"\[(\d+),(\d+)\]\[(\d+),(\d+)\]", value)
        if not match:
            raise AssertionError(f"Invalid Android bounds string: {value}")

        left, top, right, bottom = (int(group) for group in match.groups())

        return Bounds(
            x=left,
            y=top,
            width=right - left,
            height=bottom - top,
            coordinate_system=CoordinateSystem.DEVICE_PIXEL,
        )
