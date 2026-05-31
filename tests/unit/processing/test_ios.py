from __future__ import annotations

import unittest
import xml.etree.ElementTree as ET  # nosec

from fathom.constants import ActionType
from fathom.processing.parsers.ios import IOSParser


class IOSParserTest(unittest.TestCase):
    """
    Covers iOS parser scroll metadata normalization.
    """

    def test_marks_shallow_scroll_view_as_horizontal_scrollable(self) -> None:
        """
        Shallow wide scroll views should surface carousel metadata.
        """

        xml = """
        <AppiumAUT>
          <XCUIElementTypeApplication type="XCUIElementTypeApplication" x="0" y="0" width="402" height="874">
            <XCUIElementTypeScrollView type="XCUIElementTypeScrollView" x="0" y="700" width="402" height="80" visible="true" enabled="true" />
          </XCUIElementTypeApplication>
        </AppiumAUT>
        """
        root = ET.fromstring(xml)
        elements = IOSParser().find_all_elements(
            root=root, screenshot_width=1206, screenshot_height=2622
        )

        self.assertEqual(len(elements), 1)
        self.assertEqual(elements[0].attributes.get("scrollable"), "true")
        self.assertEqual(elements[0].attributes.get("axis"), "horizontal")
        self.assertEqual(elements[0].attributes.get("kind"), "carousel")

    def test_clips_partially_visible_large_textview_to_viewport(self) -> None:
        """
        iOS can report scrollable text content with document-height bounds.
        """

        xml = """
        <AppiumAUT>
          <XCUIElementTypeApplication type="XCUIElementTypeApplication" x="0" y="0" width="402" height="874">
            <XCUIElementTypeTextView type="XCUIElementTypeTextView" x="38" y="296" width="326" height="8585" visible="true" enabled="true" value="Privacy text" />
          </XCUIElementTypeApplication>
        </AppiumAUT>
        """
        root = ET.fromstring(xml)
        elements = IOSParser().find_all_elements(
            root=root, screenshot_width=1206, screenshot_height=2622
        )

        self.assertEqual(len(elements), 1)
        self.assertEqual(elements[0].bounds.x1, 38)
        self.assertEqual(elements[0].bounds.y1, 296)
        self.assertEqual(elements[0].bounds.x2, 364)
        self.assertEqual(elements[0].bounds.y2, 874)
        self.assertEqual(elements[0].attributes.get("scrollable"), "true")
        self.assertEqual(elements[0].attributes.get("kind"), "viewport")

    def test_directional_swipe_filters_to_swipeable_candidates(self) -> None:
        """
        Directional swipe actions must use the same candidate filtering as generic swipe.
        """

        xml = """
        <AppiumAUT>
          <XCUIElementTypeApplication type="XCUIElementTypeApplication" x="0" y="0" width="440" height="956">
            <XCUIElementTypeStaticText type="XCUIElementTypeStaticText" x="20" y="40" width="160" height="30" visible="true" label="More on Swiggy" />
            <XCUIElementTypeCollectionView type="XCUIElementTypeCollectionView" x="0" y="756" width="440" height="102" visible="true" enabled="true" />
          </XCUIElementTypeApplication>
        </AppiumAUT>
        """
        root = ET.fromstring(xml)
        parser = IOSParser()
        elements = parser.find_all_elements(
            root=root, screenshot_width=1320, screenshot_height=2868
        )

        filtered = parser.filter_by_action(elements=elements, action=ActionType.SWIPE_LEFT)

        self.assertLess(len(filtered), len(elements))
        self.assertTrue(filtered)
        self.assertTrue(
            all(
                element.attributes.get("type") != "XCUIElementTypeStaticText"
                for element in filtered
            )
        )
