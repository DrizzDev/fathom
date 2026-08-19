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
        self.assertEqual(elements[0].attributes.get("kind"), "list")
        self.assertEqual(elements[0].attributes.get("axis"), "vertical")
        self.assertEqual(elements[0].attributes.get("scrollable"), "true")

    def test_declares_interactivity_as_tri_state_hint(self) -> None:
        """
        Surface the hierarchy's clickable declaration as True, False, or None when absent.
        """

        xml = """
        <hierarchy package="app">
          <node class="android.widget.Button" bounds="[0,200][540,400]" clickable="true" text="Login" />
          <node class="android.widget.TextView" bounds="[0,500][540,700]" clickable="false" text="Caption" resource-id="app:id/caption" />
          <node class="android.widget.TextView" bounds="[0,800][540,1000]" text="Undeclared" resource-id="app:id/plain" />
        </hierarchy>
        """
        root = ET.fromstring(xml)
        elements = AndroidParser().find_all_elements(
            root=root, screenshot_width=1080, screenshot_height=2340
        )

        declared = {
            str(element.attributes.get("text")): element.interactive for element in elements
        }

        self.assertIsNone(declared["Undeclared"])
        self.assertIs(declared["Login"], True)
        self.assertIs(declared["Caption"], False)
