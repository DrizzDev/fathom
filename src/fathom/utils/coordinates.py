from __future__ import annotations

from typing import Tuple

from fathom.schemas.actions import BoundingBox


class CoordinateConverter:
    """
    Converts normalized coordinates to device pixels.
    Handles the translation from 0-1000 normalized scale to actual device pixel coordinates.
    """

    def __init__(self, screen_width: int, screen_height: int) -> None:
        """
        Initialize converter.

        Args:
            screen_width: Device screen width in pixels.
            screen_height: Device screen height in pixels.
        """

        self.__width = screen_width
        self.__height = screen_height

    def to_pixels(self, bbox: BoundingBox) -> Tuple[int, int, int, int]:
        """
        Convert bounding box to pixel coordinates.

        Args:
            bbox: Normalized bounding box (0-1000 scale).

        Returns:
            Tuple of (x, y, width, height) in pixels.
        """

        return bbox.to_pixels(self.__width, self.__height)

    def center_to_pixels(self, bbox: BoundingBox) -> Tuple[int, int]:
        """
        Get center point in pixel coordinates.

        Args:
            bbox: Normalized bounding box.

        Returns:
            Tuple of (x, y) center point in pixels.
        """

        x, y, w, h = self.to_pixels(bbox)
        return x + w // 2, y + h // 2

    def swipe_coordinates(
        self,
        bbox: BoundingBox,
        direction: str,
    ) -> Tuple[int, int, int, int]:
        """
        Calculate swipe start and end coordinates.

        Args:
            bbox: Target area for swipe.
            direction: One of 'up', 'down', 'left', 'right'.

        Returns:
            Tuple of (x1, y1, x2, y2) for swipe.
        """

        x, y, w, h = self.to_pixels(bbox)
        cx, cy = x + w // 2, y + h // 2

        distance_x = int(w * 0.7)
        distance_y = int(h * 0.7)

        if direction == "up":
            return cx, cy + distance_y // 2, cx, cy - distance_y // 2

        elif direction == "down":
            return cx, cy - distance_y // 2, cx, cy + distance_y // 2

        elif direction == "left":
            return cx + distance_x // 2, cy, cx - distance_x // 2, cy

        elif direction == "right":
            return cx - distance_x // 2, cy, cx + distance_x // 2, cy

        else:
            return cx, cy, cx, cy
