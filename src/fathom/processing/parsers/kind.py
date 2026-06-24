from __future__ import annotations

import xml.etree.ElementTree as ET  # nosec
from logging import getLogger
from typing import Optional, Tuple

from fathom.constants.screen import (
    GAME_SURFACE_AREA_FLOOR,
    WEBVIEW_AREA_FLOOR,
    ScreenKind,
)
from fathom.processing.parsers.geometry import ElementGeometry

logger = getLogger(__name__)


class ScreenKindClassifier:
    """
    Classifies the root hierarchy XML as native, webview, or game-surface backed.
    """

    __WEBVIEW_TAGS = (
        "android.webkit.WebView",
        "XCUIElementTypeWebView",
    )

    __GAME_SURFACE_TAGS = (
        "android.view.SurfaceView",
        "android.opengl.GLSurfaceView",
    )

    @classmethod
    def classify(
        cls,
        *,
        screen_width: int,
        screen_height: int,
        xml_content: Optional[str],
    ) -> ScreenKind:
        """
        Return the screen kind inferred from the supplied hierarchy XML.
        """

        if not xml_content or screen_width <= 0 or screen_height <= 0:
            return ScreenKind.NATIVE

        screen_area = screen_width * screen_height

        try:
            root = ET.fromstring(xml_content)  # nosec
        except ET.ParseError:
            return ScreenKind.NATIVE

        webview_ratio = cls.__largest_area_ratio(
            root=root,
            tags=cls.__WEBVIEW_TAGS,
            screen_area=screen_area,
        )
        game_surface_ratio = cls.__largest_area_ratio(
            root=root,
            screen_area=screen_area,
            tags=cls.__GAME_SURFACE_TAGS,
        )

        kind = cls.__decide_kind(
            webview_ratio=webview_ratio,
            game_surface_ratio=game_surface_ratio,
        )

        logger.info(
            "Screen kind classified",
            extra={
                "kind": kind.value,
                "event": "screen.kind.classified",
                "component": "processing.parsers.kind",
                "webview.area_ratio": round(webview_ratio, 4),
                "game_surface.area_ratio": round(game_surface_ratio, 4),
            },
        )

        return kind

    @staticmethod
    def __decide_kind(*, webview_ratio: float, game_surface_ratio: float) -> ScreenKind:
        """
        Choose the dominant kind by comparing area ratios against configured floors.
        """

        if game_surface_ratio >= GAME_SURFACE_AREA_FLOOR:
            return ScreenKind.GAME_SURFACE

        if webview_ratio >= WEBVIEW_AREA_FLOOR:
            return ScreenKind.WEBVIEW

        return ScreenKind.NATIVE

    @classmethod
    def __largest_area_ratio(
        cls,
        *,
        root: ET.Element,
        screen_area: int,
        tags: Tuple[str, ...],
    ) -> float:
        """
        Return the largest matching element's area as a fraction of total screen area.
        """

        if screen_area <= 0:
            return 0.0

        largest_area = 0

        for tag in tags:
            for node in root.findall(f".//{tag}"):
                largest_area = max(largest_area, cls.__area(element=node))

            for node in root.findall(f".//*[@class='{tag}']"):
                largest_area = max(largest_area, cls.__area(element=node))

        return largest_area / screen_area

    @staticmethod
    def __area(*, element: ET.Element) -> int:
        """
        Return the element's pixel area, or zero when it has no usable geometry.
        """

        rect = ElementGeometry.rect_of(element=element)
        return rect.area if rect is not None else 0
