from __future__ import annotations

from typing import Optional, Tuple

from fathom.schemas.actions import Bounds, CoordinateSource, ExecutionRegion, GesturePath
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

        For normalized coordinates, this preserves the gold-standard ambiguity
        handling when VLM outputs (x, y) as either top-left or center.
        """

        if bounds.system == "pixel" or not bounds.is_normalized:
            x, y, width, height = bounds.to_pixels(
                screen_width=self.__width, screen_height=self.__height
            )
            return x + width // 2, y + height // 2

        x_px, y_px, width_px, height_px = bounds.to_pixels(
            screen_width=self.__width, screen_height=self.__height
        )
        # Clamp to the last on-screen pixel (0..width-1 / 0..height-1).
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
        Return swipe coordinates derived from the bounds edges.
        """

        region = self.region_from_bounds(bounds=bounds, source="model")
        return self.resolve_swipe_path(region=region, direction=direction).to_coordinates()

    def region_from_bounds(self, *, bounds: Bounds, source: CoordinateSource) -> ExecutionRegion:
        """
        Convert action bounds into an execution region in screen pixels.
        """

        x, y, width, height = self.to_pixels(bounds=bounds)
        return ExecutionRegion(x=x, y=y, width=width, height=height, source=source)

    def viewport_region(self) -> ExecutionRegion:
        """
        Return the full screen as a viewport execution region.
        """

        return ExecutionRegion(
            x=0,
            y=0,
            source="viewport",
            width=self.__width,
            height=self.__height,
        )

    def resolve_swipe_path(self, *, region: ExecutionRegion, direction: str) -> GesturePath:
        """
        Derive a finger swipe path from the execution region edges.
        """

        swipe_policy = self.__configuration.interaction.policy.swipe

        horizontal_margin = self.__edge_margin(
            size=region.width,
            ratio=swipe_policy.edge_margin_ratio,
            minimum=swipe_policy.minimum_edge_margin,
            maximum=swipe_policy.maximum_edge_margin,
        )
        vertical_margin = self.__edge_margin(
            size=region.height,
            ratio=swipe_policy.edge_margin_ratio,
            minimum=swipe_policy.minimum_edge_margin,
            maximum=swipe_policy.maximum_edge_margin,
        )
        center_x = region.x + region.width // 2
        center_y = region.y + region.height // 2

        if direction == "up":
            return GesturePath(
                start_x=center_x,
                start_y=region.y + region.height - vertical_margin,
                end_x=center_x,
                end_y=region.y + vertical_margin,
                duration=swipe_policy.duration,
            )

        if direction == "down":
            return GesturePath(
                start_x=center_x,
                start_y=region.y + vertical_margin,
                end_x=center_x,
                end_y=region.y + region.height - vertical_margin,
                duration=swipe_policy.duration,
            )

        if direction == "left":
            return GesturePath(
                start_x=region.x + region.width - horizontal_margin,
                start_y=center_y,
                end_x=region.x + horizontal_margin,
                end_y=center_y,
                duration=swipe_policy.duration,
            )

        if direction == "right":
            return GesturePath(
                start_x=region.x + horizontal_margin,
                start_y=center_y,
                end_x=region.x + region.width - horizontal_margin,
                end_y=center_y,
                duration=swipe_policy.duration,
            )

        return GesturePath(
            start_x=center_x,
            start_y=center_y,
            end_x=center_x,
            end_y=center_y,
            duration=swipe_policy.duration,
        )

    def resolve_scroll_path(self, *, region: ExecutionRegion, direction: str) -> GesturePath:
        """
        Derive a content scroll path from the viewport edges.
        """

        swipe_policy = self.__configuration.interaction.policy.swipe
        scroll_policy = self.__configuration.interaction.policy.scroll

        vertical_margin = self.__edge_margin(
            size=region.height,
            ratio=scroll_policy.edge_margin_ratio,
            minimum=scroll_policy.minimum_edge_margin,
            maximum=scroll_policy.maximum_edge_margin,
        )
        center_x = region.x + region.width // 2

        if direction == "up":
            return GesturePath(
                start_x=center_x,
                start_y=region.y + vertical_margin,
                end_x=center_x,
                end_y=region.y + region.height - vertical_margin,
                duration=swipe_policy.duration,
            )

        return GesturePath(
            start_x=center_x,
            start_y=region.y + region.height - vertical_margin,
            end_x=center_x,
            end_y=region.y + vertical_margin,
            duration=swipe_policy.duration,
        )

    @staticmethod
    def __edge_margin(*, size: int, ratio: float, minimum: int, maximum: int) -> int:
        """
        Return a safe margin that preserves travel across small regions.
        """

        if size <= 1:
            return 0

        preferred = int(size * ratio)
        midpoint_limit = max(0, (size - 1) // 2)
        bounded = max(minimum, min(preferred, maximum))

        return min(bounded, midpoint_limit)
