from __future__ import annotations

import unittest

from fathom.constants.screen import ScreenKind
from fathom.processing.parsers.kind import ScreenKindClassifier


class ScreenKindClassifierTest(unittest.TestCase):
    """
    Locks the area-ratio classification after the shared-geometry refactor.
    """

    def test_dominant_webview_is_classified_as_webview(self) -> None:
        xml = (
            "<hierarchy>"
            '<node class="android.webkit.WebView" bounds="[0,0][1000,1800]"/>'
            "</hierarchy>"
        )
        kind = ScreenKindClassifier.classify(screen_width=1000, screen_height=2000, xml_content=xml)
        self.assertEqual(kind, ScreenKind.WEBVIEW)

    def test_small_webview_stays_native(self) -> None:
        xml = (
            '<hierarchy><node class="android.webkit.WebView" bounds="[0,0][100,100]"/></hierarchy>'
        )
        kind = ScreenKindClassifier.classify(screen_width=1000, screen_height=2000, xml_content=xml)
        self.assertEqual(kind, ScreenKind.NATIVE)

    def test_absent_hierarchy_is_native(self) -> None:
        kind = ScreenKindClassifier.classify(
            screen_width=1000, screen_height=2000, xml_content=None
        )
        self.assertEqual(kind, ScreenKind.NATIVE)


if __name__ == "__main__":
    unittest.main()
