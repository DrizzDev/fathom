from __future__ import annotations

from logging import getLogger
from typing import Literal, Optional, Tuple

from fathom.schemas.actions import Bounds
from fathom.schemas.configuration import DeviceRuntimeConfiguration

logger = getLogger(__name__)


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

        Endpoints are clamped to a safe in-screen region (defined by
        ``edge_margin_ratio`` on the relevant interaction policy) so
        gestures never start or end on the screen boundary, where iOS
        treats them as back/app-switch swipes and Android silently
        clamps them. If edge clamping shrinks the gesture below
        ``min_distance_px``, the swipe is recentered on the screen
        midline along the swipe axis to recover a usable distance.
        """

        _, _, width, height = self.to_pixels(bounds=bounds)
        center_x, center_y = self.center_to_pixels(bounds=bounds)

        swipe_policy = self.__configuration.interaction.policy.swipe
        scroll_policy = self.__configuration.interaction.policy.scroll

        distance_x = max(450, int(width * swipe_policy.distance_ratio))
        distance_y = max(450, int(height * scroll_policy.distance_ratio))

        if direction in ("up", "down"):
            return self.__build_axial_swipe(
                axis="y",
                fixed=center_x,
                center=center_y,
                distance=distance_y,
                reverse=(direction == "up"),
                edge_margin_ratio=scroll_policy.edge_margin_ratio,
                min_distance_px=scroll_policy.min_distance_px,
            )

        if direction in ("left", "right"):
            return self.__build_axial_swipe(
                axis="x",
                fixed=center_y,
                center=center_x,
                distance=distance_x,
                reverse=(direction == "left"),
                edge_margin_ratio=swipe_policy.edge_margin_ratio,
                min_distance_px=swipe_policy.min_distance_px,
            )

        logger.warning(
            "Unknown swipe direction %r; falling back to vertical midline scroll-up",
            direction,
        )
        return self.__build_axial_swipe(
            axis="y",
            fixed=self.__width // 2,
            center=self.__height // 2,
            distance=distance_y,
            reverse=True,
            edge_margin_ratio=scroll_policy.edge_margin_ratio,
            min_distance_px=scroll_policy.min_distance_px,
        )

    def __build_axial_swipe(
        self,
        *,
        axis: Literal["x", "y"],
        fixed: int,
        center: int,
        distance: int,
        reverse: bool,
        edge_margin_ratio: float,
        min_distance_px: int,
    ) -> Tuple[int, int, int, int]:
        """
        Build a swipe along one axis with edge clamping and min-distance
        midline recovery.

        ``fixed`` is the constant coordinate on the perpendicular axis
        (e.g. the x coordinate of a vertical swipe). ``center`` is the
        midpoint along the swipe axis. ``reverse=True`` means the start
        coordinate is greater than the end (e.g. swipe-up starts below
        center, ends above).
        """

        screen_dim = self.__height if axis == "y" else self.__width
        fixed_screen_dim = self.__width if axis == "y" else self.__height

        margin = max(0, int(screen_dim * edge_margin_ratio))
        safe_low = margin
        safe_high = max(safe_low, screen_dim - 1 - margin)
        safe_span = safe_high - safe_low

        half = distance // 2
        if reverse:
            start, end = center + half, center - half
        else:
            start, end = center - half, center + half

        start = max(safe_low, min(start, safe_high))
        end = max(safe_low, min(end, safe_high))

        recovery_threshold = min(min_distance_px, safe_span)
        if abs(end - start) < recovery_threshold:
            desired = min(max(distance, min_distance_px), safe_span)
            midline = (safe_low + safe_high) // 2
            half_desired = desired // 2
            if reverse:
                start, end = midline + half_desired, midline - half_desired
            else:
                start, end = midline - half_desired, midline + half_desired
            start = max(safe_low, min(start, safe_high))
            end = max(safe_low, min(end, safe_high))

        fixed_clamped = max(0, min(fixed, fixed_screen_dim - 1))

        if axis == "y":
            return fixed_clamped, start, fixed_clamped, end
        return start, fixed_clamped, end, fixed_clamped
