from __future__ import annotations

import unittest
from pathlib import Path

from fathom.constants.screen import (
    GAME_SURFACE_AREA_FLOOR,
    WEBVIEW_AREA_FLOOR,
    ScreenKind,
)
from fathom.processing.parsers.kind import ScreenKindClassifier

_FULL_SCREEN_WEBVIEW_FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "processing"
    / "kind"
    / "webview_full_screen.xml"
)


class ScreenKindClassifierTest(unittest.TestCase):
    """
    Pins :meth:`ScreenKindClassifier.classify` across native, webview, and game-surface frames.
    """

    def test_empty_xml_returns_native(self) -> None:
        """
        Missing hierarchy degrades safely to NATIVE rather than mis-routing OCR.
        """

        result = ScreenKindClassifier.classify(
            xml_content=None,
            screen_width=2340,
            screen_height=1080,
        )

        self.assertIs(result, ScreenKind.NATIVE)

    def test_zero_dimensions_returns_native(self) -> None:
        """
        Zero / negative dimensions short-circuit to NATIVE; no divide-by-zero risk.
        """

        result = ScreenKindClassifier.classify(
            screen_width=0,
            screen_height=1080,
            xml_content="<root/>",
        )

        self.assertIs(result, ScreenKind.NATIVE)

    def test_malformed_xml_returns_native(self) -> None:
        """
        Parse failures degrade to NATIVE — classifier never raises into the perception layer.
        """

        result = ScreenKindClassifier.classify(
            screen_width=2340,
            screen_height=1080,
            xml_content="<<<not-valid-xml",
        )

        self.assertIs(result, ScreenKind.NATIVE)

    def test_simple_button_hierarchy_is_native(self) -> None:
        """
        A tappable button alone classifies as NATIVE.
        """

        xml = '<root><android.widget.Button bounds="[10,10][50,50]"/></root>'

        result = ScreenKindClassifier.classify(
            xml_content=xml,
            screen_width=2340,
            screen_height=1080,
        )

        self.assertIs(result, ScreenKind.NATIVE)

    def test_appium_tag_webview_above_floor_returns_webview(self) -> None:
        """
        An Appium-style WebView tag covering >= the floor area classifies as WEBVIEW.
        """

        xml = '<root><android.webkit.WebView bounds="[0,0][2340,1080]"/></root>'

        result = ScreenKindClassifier.classify(
            xml_content=xml,
            screen_width=2340,
            screen_height=1080,
        )

        self.assertIs(result, ScreenKind.WEBVIEW)

    def test_uiautomator_class_attribute_webview_returns_webview(self) -> None:
        """
        uiautomator emits class as an attribute, not a tag; classifier must match both shapes.
        """

        xml = '<hierarchy><node class="android.webkit.WebView" bounds="[0,0][2340,1080]"/></hierarchy>'

        result = ScreenKindClassifier.classify(
            xml_content=xml,
            screen_width=2340,
            screen_height=1080,
        )

        self.assertIs(result, ScreenKind.WEBVIEW)

    def test_webview_below_floor_returns_native(self) -> None:
        """
        A WebView whose area is below WEBVIEW_AREA_FLOOR must not flip the classification.
        """

        small_height = int(2340 * 1080 * (WEBVIEW_AREA_FLOOR / 2.0) / 2340)
        xml = f'<root><android.webkit.WebView bounds="[0,0][2340,{small_height}]"/></root>'

        result = ScreenKindClassifier.classify(
            xml_content=xml,
            screen_width=2340,
            screen_height=1080,
        )

        self.assertIs(result, ScreenKind.NATIVE)

    def test_full_screen_surfaceview_returns_game_surface(self) -> None:
        """
        A SurfaceView covering >= the game-surface floor area classifies as GAME_SURFACE.
        """

        xml = '<root><android.view.SurfaceView bounds="[0,0][2340,1080]"/></root>'

        result = ScreenKindClassifier.classify(
            xml_content=xml,
            screen_width=2340,
            screen_height=1080,
        )

        self.assertIs(result, ScreenKind.GAME_SURFACE)

    def test_ios_webview_tag_returns_webview(self) -> None:
        """
        iOS Appium uses ``XCUIElementTypeWebView`` with x/y/width/height attributes.
        """

        xml = '<root><XCUIElementTypeWebView x="0" y="0" width="1170" height="2532"/></root>'

        result = ScreenKindClassifier.classify(
            xml_content=xml,
            screen_width=1170,
            screen_height=2532,
        )

        self.assertIs(result, ScreenKind.WEBVIEW)

    def test_game_surface_floor_above_webview_floor_short_circuits(self) -> None:
        """
        Floor invariant: GAME_SURFACE_AREA_FLOOR must be no smaller than WEBVIEW_AREA_FLOOR
        so a full-screen surface can never resolve as a webview.
        """

        self.assertGreaterEqual(GAME_SURFACE_AREA_FLOOR, WEBVIEW_AREA_FLOOR)

    @unittest.skipUnless(
        _FULL_SCREEN_WEBVIEW_FIXTURE.exists(),
        "full-screen webview hierarchy fixture not present on disk",
    )
    def test_full_screen_webview_hierarchy_fixture_classifies_as_webview(self) -> None:
        """
        End-to-end pin: a real captured WebView hierarchy still classifies as WEBVIEW.
        """

        xml = _FULL_SCREEN_WEBVIEW_FIXTURE.read_text()

        result = ScreenKindClassifier.classify(
            xml_content=xml,
            screen_width=2340,
            screen_height=1080,
        )

        self.assertIs(result, ScreenKind.WEBVIEW)
