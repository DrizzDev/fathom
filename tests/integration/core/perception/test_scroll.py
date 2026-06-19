from __future__ import annotations

import unittest
import xml.etree.ElementTree as ET  # nosec
from pathlib import Path

from PIL import Image

from fathom.constants import ActionType
from fathom.core.perception.observation import ScreenObservationService
from fathom.processing.parsers.ios import IOSParser
from fathom.schemas.actions import CoordinateSource, CoordinateSystem
from fathom.schemas.budgets import PerceptionBudget
from fathom.schemas.screens import ScreenCapture, ScreenHashBundle
from fathom.utils.coordinates import CoordinateConverter


class ScrollRegionCoordinateRegressionTest(unittest.IsolatedAsyncioTestCase):
    """
    Replays captured production-style fixtures through scroll-region perception.
    """

    __FIXTURES = Path(__file__).resolve().parents[3] / "fixtures" / "perception" / "scroll"

    @staticmethod
    def __hashes() -> ScreenHashBundle:
        """
        Return deterministic hashes for observation-only tests.
        """

        return ScreenHashBundle(
            visual_hash="0" * 16,
            xml_hash="a" * 16,
            interaction_hash="b" * 16,
        )

    async def test_logical_capture_dimensions_preserve_page_region_scale(self) -> None:
        """
        Logical capture dimensions with a larger PNG must not be treated as device pixels.
        """

        xml_path = self.__FIXTURES / "AEKCI" / "hierarchy.xml"
        screenshot_path = self.__FIXTURES / "AEKCI" / "screenshot.png"
        xml = xml_path.read_text(encoding="utf-8")
        root = ET.fromstring(xml)
        app = root.find(".//XCUIElementTypeApplication")
        self.assertIsNotNone(app)
        assert app is not None

        with Image.open(screenshot_path) as screenshot:
            pixel_width, pixel_height = screenshot.size
        logical_width = int(app.get("width", "0"))
        logical_height = int(app.get("height", "0"))
        elements = IOSParser().find_all_elements(
            root=root,
            screenshot_width=pixel_width,
            screenshot_height=pixel_height,
        )

        observation = await ScreenObservationService().observe(
            capture=ScreenCapture(
                width=logical_width,
                height=logical_height,
                activity="bundl.swiggy",
                image=screenshot_path.read_bytes(),
                xml_content=xml,
                timestamp=1780051837410,
            ),
            hashes=self.__hashes(),
            budget=PerceptionBudget(ocr=0, local=0, localization=0),
            manifest=tuple(elements),
            session_id="logical-capture-regression",
            step_number=4,
        )

        self.assertTrue(observation.scroll)
        region = observation.scroll[0]
        self.assertEqual(region.bounds.system, CoordinateSystem.LOGICAL)

        converter = CoordinateConverter(
            logical_width=logical_width,
            logical_height=logical_height,
            pixel_width=pixel_width,
            pixel_height=pixel_height,
        )
        execution_region = converter.region_from_bounds(
            bounds=region.bounds,
            source=region.bounds.source or CoordinateSource.VIEWPORT,
        )
        self.assertGreaterEqual(execution_region.width, int(logical_width * 0.90))

    def test_directional_swipe_filter_retains_scrollable_collection_view(self) -> None:
        """
        Directional swipe filtering should keep scrollable collection views, not static text.
        """

        xml_path = self.__FIXTURES / "FsIfE" / "hierarchy.xml"
        root = ET.fromstring(xml_path.read_text(encoding="utf-8"))
        parser = IOSParser()
        elements = parser.find_all_elements(
            root=root,
            screenshot_width=1320,
            screenshot_height=2868,
        )
        filtered = parser.filter_by_action(elements=elements, action=ActionType.SWIPE_LEFT)

        self.assertLess(len(filtered), len(elements))
        self.assertTrue(
            any(
                element.attributes.get("type") == "XCUIElementTypeCollectionView"
                for element in filtered
            )
        )
        self.assertFalse(
            any(
                element.attributes.get("type") == "XCUIElementTypeStaticText"
                for element in filtered
            )
        )
