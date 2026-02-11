from __future__ import annotations

from typing import Tuple

from fathom.schemas.actions import Bounds


class CoordinateConverter:
    """
    Converts normalized coordinates to device pixels.
    """

    def __init__(self, screen_width: int, screen_height: int) -> None:
        """
        Initialize converter.
        """

        self.__width = screen_width
        self.__height = screen_height

    def to_pixels(self, bounds: Bounds) -> Tuple[int, int, int, int]:
        """
        Convert bounding box to pixel coordinates.
        """

        return bounds.to_pixels(screen_width=self.__width, screen_height=self.__height)

    def center_to_pixels(self, bounds: Bounds) -> Tuple[int, int]:
        """
        Get center point in pixel coordinates.
        """

        x, y, width, height = self.to_pixels(bounds=bounds)
        return x + width // 2, y + height // 2

    def swipe_coordinates(
        self,
        bounds: Bounds,
        direction: str,
    ) -> Tuple[int, int, int, int]:
        """
        Calculate swipe start and end coordinates.
        """

        x, y, width, height = self.to_pixels(bounds=bounds)
        center_x, center_y = x + width // 2, y + height // 2

        distance_x = int(width * 0.7)
        distance_y = int(height * 0.2)

        if direction == "up":
            return center_x, center_y + distance_y // 2, center_x, center_y - distance_y // 2

        elif direction == "down":
            return center_x, center_y - distance_y // 2, center_x, center_y + distance_y // 2

        elif direction == "left":
            return center_x + distance_x // 2, center_y, center_x - distance_x // 2, center_y

        elif direction == "right":
            return center_x - distance_x // 2, center_y, center_x + distance_x // 2, center_y

        else:
            return center_x, center_y, center_x, center_y
