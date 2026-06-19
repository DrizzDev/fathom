from __future__ import annotations

import unittest
import xml.etree.ElementTree as ET  # nosec

from fathom.processing.parsers.android import AndroidParser


class AndroidParserTest(unittest.TestCase):
    """
    Covers Android parser scroll metadata normalization.
    """

    def test_marks_recycler_view_as_vertical_scrollable(self) -> None:
        """
        RecyclerView nodes should surface normalized vertical scroll metadata.
        """

        xml = """
        <hierarchy package="app">
          <node class="androidx.recyclerview.widget.RecyclerView" bounds="[0,200][1080,2200]" scrollable="true" clickable="false" text="" />
        </hierarchy>
        """
        root = ET.fromstring(xml)
        elements = AndroidParser().find_all_elements(
            root=root, screenshot_width=1080, screenshot_height=2340
        )

        self.assertEqual(len(elements), 1)
        self.assertEqual(elements[0].attributes.get("scrollable"), "true")
        self.assertEqual(elements[0].attributes.get("axis"), "vertical")
        self.assertEqual(elements[0].attributes.get("kind"), "list")
