"""
Hit-testing a tap coordinate against the interactive elements of a hierarchy.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET  # nosec
from typing import Iterator, Optional

from fathom.constants.screen import HitOutcome
from fathom.processing.parsers.geometry import ElementGeometry, PixelRect


class InteractiveHitTester:
    """
    Decides whether a tap coordinate fell on an interactive hierarchy element.
    """

    __TRUE = "true"
    __FALSE = "false"
    __INTERACTIVE_ATTRS = ("clickable", "long-clickable", "focusable", "checkable")

    @classmethod
    def locate(cls, *, xml_content: Optional[str], point_x: int, point_y: int) -> HitOutcome:
        """
        Returns HIT/MISS for the point, or UNKNOWN when the hierarchy cannot judge it.

        UNKNOWN covers an absent or unparseable hierarchy and one that exposes no
        interactive elements at all (a WebView surface or a non-Android tree), so a
        screen we cannot reason about never downgrades a defect.
        """

        if not xml_content:
            return HitOutcome.UNKNOWN

        try:
            root = ET.fromstring(xml_content)  # nosec
        except ET.ParseError:
            return HitOutcome.UNKNOWN

        seen_interactive = False
        for rect in cls.__interactive_rects(root=root):
            seen_interactive = True
            if rect.contains(x=point_x, y=point_y):
                return HitOutcome.HIT

        if not seen_interactive:
            return HitOutcome.UNKNOWN

        return HitOutcome.MISS

    @classmethod
    def __interactive_rects(cls, *, root: ET.Element) -> Iterator[PixelRect]:
        """
        Yields the rectangle of every enabled, interactive element in the tree.
        """

        for element in root.iter():
            if not cls.__is_interactive(element=element):
                continue
            rect = ElementGeometry.rect_of(element=element)
            if rect is not None and rect.area > 0:
                yield rect

    @classmethod
    def __is_interactive(cls, *, element: ET.Element) -> bool:
        """
        Whether an element is enabled and accepts taps per its hierarchy flags.
        """

        if element.get("enabled", cls.__TRUE) == cls.__FALSE:
            return False
        return any(element.get(attr) == cls.__TRUE for attr in cls.__INTERACTIVE_ATTRS)
