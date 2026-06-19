from __future__ import annotations

import unittest
import xml.etree.ElementTree as ET  # nosec
from pathlib import Path

import pytest

from fathom.constants import ActionType
from fathom.processing.parsers.ios import IOSParser

pytestmark = pytest.mark.release


class LiveScrollRegionFixtureTest(unittest.TestCase):
    """
    Release-gated smoke tests over captured scroll-region fixtures.
    """

    __ROOT = Path(__file__).resolve().parents[4]
    __DEBUG = __ROOT / "debug" / "scroll"

    def test_directional_swipe_filter_excludes_static_text(self) -> None:
        """
        Directional swipe filtering must not treat static labels as gesture targets.
        """

        xml_path = self.__DEBUG / "FsIfE" / "assets" / "xmls" / "1780051090775__bundl.swiggy.png"
        root = ET.fromstring(xml_path.read_text(encoding="utf-8"))
        parser = IOSParser()
        elements = parser.find_all_elements(
            root=root,
            screenshot_width=1320,
            screenshot_height=2868,
        )

        filtered = parser.filter_by_action(elements=elements, action=ActionType.SWIPE_LEFT)

        self.assertTrue(filtered)
        self.assertFalse(
            any(
                element.attributes.get("type") == "XCUIElementTypeStaticText"
                for element in filtered
            )
        )
