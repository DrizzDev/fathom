"""
Shared hierarchy geometry parsing for the screen-hierarchy classifiers.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Optional

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    import xml.etree.ElementTree as ET  # nosec


class PixelRect(BaseModel):
    """
    An axis-aligned rectangle in device-pixel space.
    """

    model_config = ConfigDict(frozen=True)

    left: int = Field(description="Left edge in device pixels")
    top: int = Field(description="Top edge in device pixels")
    right: int = Field(description="Right edge in device pixels")
    bottom: int = Field(description="Bottom edge in device pixels")

    @property
    def area(self) -> int:
        """
        Pixel area of the rectangle, clamped so inverted extents read as zero.
        """

        return max(0, self.right - self.left) * max(0, self.bottom - self.top)

    def contains(self, *, x: int, y: int) -> bool:
        """
        Whether a point falls within the rectangle, inclusive of its edges.
        """

        return self.left <= x <= self.right and self.top <= y <= self.bottom


class ElementGeometry:
    """
    Reads an element's pixel rectangle from Android bounds or iOS frame attributes.
    """

    __BOUNDS_PATTERN = re.compile(r"\[(\d+),(\d+)\]\[(\d+),(\d+)\]")

    @classmethod
    def rect_of(cls, *, element: ET.Element) -> Optional[PixelRect]:
        """
        Returns the element's rectangle, or None when no usable geometry is present.
        """

        bounds_attr = element.get("bounds", "")
        if bounds_attr:
            match = cls.__BOUNDS_PATTERN.match(bounds_attr)
            if match:
                left, top, right, bottom = (int(value) for value in match.groups())
                return PixelRect(left=left, top=top, right=right, bottom=bottom)

        try:
            left = int(element.get("x", "0"))
            top = int(element.get("y", "0"))
            width = int(element.get("width", "0"))
            height = int(element.get("height", "0"))
        except ValueError:
            return None

        return PixelRect(left=left, top=top, right=left + width, bottom=top + height)
