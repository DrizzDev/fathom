from __future__ import annotations

from typing import Tuple

from fathom.schemas.actions import Bounds
from fathom.schemas.configuration import ADBConfiguration


class CoordinateConverter:
    """
    Converts normalized coordinates to device pixels.
    """

    def __init__(
        self,
        screen_width: int,
        screen_height: int,
        configuration: ADBConfiguration = ADBConfiguration(),
    ) -> None:
        """
        Initialize converter.
        """

        self.__width = screen_width
        self.__height = screen_height
        self.__configuration = configuration

    def to_pixels(self, bounds: Bounds) -> Tuple[int, int, int, int]:
        """
        Convert bounding box to pixel coordinates.
        """

        return bounds.to_pixels(screen_width=self.__width, screen_height=self.__height)

    def center_to_pixels(self, bounds: Bounds) -> Tuple[int, int]:
        """
        Get center point in pixel coordinates.

        For normalized coordinates, this preserves the gold-standard ambiguity
        handling when VLM outputs (x, y) as either top-left or center.
        """

        if bounds.system == "pixel" or not bounds.is_normalized:
            x, y, width, height = bounds.to_pixels(
                screen_width=self.__width, screen_height=self.__height
            )
            return x + width // 2, y + height // 2

        x, y, width, height = bounds.x, bounds.y, bounds.width, bounds.height

        center_x_norm: float
        center_y_norm: float

        if x + width > 1000:
            center_x_norm = x
        elif x - width / 2 < 0:
            center_x_norm = x + width / 2
        else:
            center_x_norm = x + width / 4

        if y + height > 1000:
            center_y_norm = y
        elif y - height / 2 < 0:
            center_y_norm = y + height / 2
        else:
            center_y_norm = y + height / 4

        center_x = max(0, min(int(center_x_norm * self.__width / 1000), self.__width))
        center_y = max(0, min(int(center_y_norm * self.__height / 1000), self.__height))

        return center_x, center_y

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

        distance_x = int(width * self.__configuration.swipe_distance)
        distance_y = int(height * self.__configuration.scroll_distance)

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
