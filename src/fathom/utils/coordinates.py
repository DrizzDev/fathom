from __future__ import annotations

from typing import Optional, Tuple

from fathom.schemas.actions import Bounds
from fathom.schemas.configuration import DeviceRuntimeConfiguration


class CoordinateConverter:
    """
    Converts normalized coordinates to device pixels.
    """

    def __init__(
        self,
        screen_width: int,
        screen_height: int,
        configuration: Optional[DeviceRuntimeConfiguration] = None,
    ) -> None:
        """
        Initialize converter.
        """

        self.__width = screen_width
        self.__height = screen_height
        self.__configuration = configuration or DeviceRuntimeConfiguration()

    def to_pixels(self, bounds: Bounds) -> Tuple[int, int, int, int]:
        """
        Convert bounding box to pixel coordinates.
        """

        return bounds.to_pixels(screen_width=self.__width, screen_height=self.__height)

    def center_to_pixels(self, bounds: Bounds) -> Tuple[int, int]:
        """
        Get center point in pixel coordinates.

        When bounds have ``width == 0 and height == 0`` (center-point format),
        (x, y) IS the center — scale directly with no heuristic.

        For bbox format (width/height > 0, e.g. label-snapped pixel bounds),
        computes the geometric center of the bounding box.
        """

        if bounds.system == "pixel" or not bounds.is_normalized:
            x_px, y_px, w_px, h_px = self.to_pixels(bounds=bounds)
            return x_px + w_px // 2, y_px + h_px // 2

        x, y, w, h = bounds.x, bounds.y, bounds.width, bounds.height

        # Center-point format: x, y IS the center already.
        if w == 0 and h == 0:
            cx = max(0, min(int(x * self.__width / 1000), self.__width - 1))
            cy = max(0, min(int(y * self.__height / 1000), self.__height - 1))
            return cx, cy

        # Bounding box format (label-snapped): compute center from bbox
        x_px, y_px, width_px, height_px = bounds.to_pixels(
            screen_width=self.__width, screen_height=self.__height
        )
        max_x = max(0, self.__width - 1)
        max_y = max(0, self.__height - 1)
        center_x = max(0, min(x_px + width_px // 2, max_x))
        center_y = max(0, min(y_px + height_px // 2, max_y))

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
        center_x, center_y = self.center_to_pixels(bounds=bounds)

        swipe_policy = self.__configuration.interaction.policy.swipe
        scroll_policy = self.__configuration.interaction.policy.scroll

        distance_x = max(450, int(width * swipe_policy.distance_ratio))
        distance_y = max(450, int(height * scroll_policy.distance_ratio))

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
