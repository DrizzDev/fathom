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

        When bounds have ``width == 0 and height == 0`` (center-point format
        from ``tap_target``), (x, y) IS the center — scale directly with
        no heuristic.

        For legacy bbox format (width/height > 0), falls back to a w/4
        heuristic to handle Gemini's inconsistent top-left vs center
        coordinate output.
        """

        if not bounds.is_normalized:
            x_px, y_px, w_px, h_px = self.to_pixels(bounds=bounds)
            return x_px + w_px // 2, y_px + h_px // 2

        x, y, w, h = bounds.x, bounds.y, bounds.width, bounds.height

        # Center-point format (tap_target): x, y IS the center already.
        if w == 0 and h == 0:
            cx = max(0, min(int(x * self.__width / 1000), self.__width))
            cy = max(0, min(int(y * self.__height / 1000), self.__height))
            return cx, cy

        # Legacy bbox format: apply disambiguation heuristic
        cx_norm: float
        if x + w > 1000:
            cx_norm = x
        elif x - w / 2 < 0:
            cx_norm = x + w / 2
        else:
            cx_norm = x + w / 4

        cy_norm: float
        if y + h > 1000:
            cy_norm = y
        elif y - h / 2 < 0:
            cy_norm = y + h / 2
        else:
            cy_norm = y + h / 4

        cx = max(0, min(int(cx_norm * self.__width / 1000), self.__width))
        cy = max(0, min(int(cy_norm * self.__height / 1000), self.__height))

        return cx, cy

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

        distance_x = int(width * 0.7)
        distance_y = int(height * 0.45)

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
