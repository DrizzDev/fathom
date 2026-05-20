from __future__ import annotations

from logging import getLogger
from typing import Optional, Tuple

from fathom.schemas.actions import (
    Bounds,
    CoordinateSource,
    ExecutionRegion,
    GesturePath,
)
from fathom.schemas.configuration import DeviceRuntimeConfiguration

logger = getLogger(__name__)


class CoordinateConverter:
    """
    Translate :class:`Bounds` from any coordinate system to logical dispatch coordinates.
    """

    def __init__(
        self,
        *,
        logical_width: int,
        logical_height: int,
        pixel_width: Optional[int] = None,
        pixel_height: Optional[int] = None,
        configuration: Optional[DeviceRuntimeConfiguration] = None,
        workflow_id: Optional[str] = None,
    ) -> None:
        """
        Bind logical (dispatch-space) and pixel (screenshot-space) dimensions.
        """

        self.__workflow_id = workflow_id
        self.__logical_width = logical_width
        self.__logical_height = logical_height
        self.__pixel_width = pixel_width or logical_width
        self.__pixel_height = pixel_height or logical_height
        self.__configuration = configuration or DeviceRuntimeConfiguration()

        logger.info(
            "CoordinateConverter initialised",
            extra={
                "workflow.id": workflow_id,
                "component": "utils.coordinates",
                "event": "converter.initialised",
                "pixel.width": self.__pixel_width,
                "pixel.height": self.__pixel_height,
                "logical.width": self.__logical_width,
                "logical.height": self.__logical_height,
                "scale.x": round(self.__pixel_width / max(1, self.__logical_width), 4),
                "scale.y": round(self.__pixel_height / max(1, self.__logical_height), 4),
            },
        )

    @property
    def logical_width(self) -> int:
        """
        Width of the dispatch (logical) coordinate space.
        """

        return self.__logical_width

    @property
    def logical_height(self) -> int:
        """
        Height of the dispatch (logical) coordinate space.
        """

        return self.__logical_height

    def to_pixels(self, bounds: Bounds) -> Tuple[int, int, int, int]:
        """
        Return ``bounds`` translated into logical dispatch coordinates.

        Method name retained for call-site compatibility;
        the returned coordinates are always in logical (appium-dispatch) space.
        """

        return bounds.to_logical_dispatch(
            pixel_width=self.__pixel_width,
            pixel_height=self.__pixel_height,
            logical_width=self.__logical_width,
            logical_height=self.__logical_height,
        )

    def center_to_pixels(self, bounds: Bounds) -> Tuple[int, int]:
        """
        Return the bounds' center in logical dispatch coordinates.
        """

        x, y, width, height = self.to_pixels(bounds=bounds)
        center_x, center_y = x + width // 2, y + height // 2

        logger.debug(
            "CoordinateConverter.center_to_pixels",
            extra={
                "component": "utils.coordinates",
                "workflow.id": self.__workflow_id,
                "event": "converter.dispatch.center",
                "source.system": bounds.system.value,
                "source.bounds": {
                    "x": bounds.x,
                    "y": bounds.y,
                    "width": bounds.width,
                    "height": bounds.height,
                },
                "logical.dispatched": {
                    "x": center_x,
                    "y": center_y,
                    "width": width,
                    "height": height,
                },
            },
        )
        return center_x, center_y

    def swipe_coordinates(
        self,
        bounds: Bounds,
        direction: str,
    ) -> Tuple[int, int, int, int]:
        """
        Return swipe coordinates derived from the bounds edges.
        """

        region = self.region_from_bounds(bounds=bounds, source=CoordinateSource.MODEL)
        return self.resolve_swipe_path(region=region, direction=direction).to_coordinates()

    def region_from_bounds(self, *, bounds: Bounds, source: CoordinateSource) -> ExecutionRegion:
        """
        Convert action bounds into an execution region in screen pixels.
        """

        x, y, width, height = self.to_pixels(bounds=bounds)
        return ExecutionRegion(x=x, y=y, width=width, height=height, source=source)

    def viewport_region(self) -> ExecutionRegion:
        """
        Return the full screen as a viewport execution region in logical points.
        """

        return ExecutionRegion(
            x=0,
            y=0,
            width=self.__logical_width,
            height=self.__logical_height,
            source=CoordinateSource.VIEWPORT,
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
