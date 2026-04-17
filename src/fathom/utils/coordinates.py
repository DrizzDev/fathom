from __future__ import annotations

from typing import Tuple

from fathom.schemas.actions import Bounds

# Fixed swipe magnitude in device pixels. All swipe/scroll actions travel
# exactly this distance regardless of the element's bounds or the screen
# size, so behavior is deterministic and predictable. The pivot (center)
# still comes from the LLM-decided coordinates; only the distance is fixed.
#
# 350px is well above Android's touch slop (~24-32px), large enough to
# trigger a fling, and conservative enough to stay inside most viewports
# without heavy clamping.
_SWIPE_DISTANCE_PX = 350


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

        Pivot is the bounds' center — for LLM-emitted actions that is the
        ``tap_target`` point the model chose. Distance is a fixed constant
        (:data:`_SWIPE_DISTANCE_PX`) regardless of bounds size or screen
        size, so swipes are deterministic. Endpoints are clamped to the
        device viewport.
        """

        center_x, center_y = self.center_to_pixels(bounds=bounds)
        half = _SWIPE_DISTANCE_PX // 2

        def _clamp_x(val: int) -> int:
            return max(0, min(val, self.__width - 1))

        def _clamp_y(val: int) -> int:
            return max(0, min(val, self.__height - 1))

        if direction == "up":
            return (
                center_x,
                _clamp_y(center_y + half),
                center_x,
                _clamp_y(center_y - half),
            )

        if direction == "down":
            return (
                center_x,
                _clamp_y(center_y - half),
                center_x,
                _clamp_y(center_y + half),
            )

        if direction == "left":
            return (
                _clamp_x(center_x + half),
                center_y,
                _clamp_x(center_x - half),
                center_y,
            )

        if direction == "right":
            return (
                _clamp_x(center_x - half),
                center_y,
                _clamp_x(center_x + half),
                center_y,
            )

        return center_x, center_y, center_x, center_y
