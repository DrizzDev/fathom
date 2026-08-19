from __future__ import annotations

from typing import Optional

from fathom.schemas.actions import Bounds, GesturePath
from fathom.schemas.swipe import ReservePolicy


class AnchorGuard:
    """
    Confines gesture touch-down points to the viewport area the operating system leaves addressable.

    Mobile platforms route a gesture to their own navigation handlers based on
    where the pointer goes down rather than where it lifts, so only the
    touch-down point is confined and gesture travel is preserved.
    """

    def addressable(self, *, viewport: Bounds, policy: ReservePolicy) -> Optional[Bounds]:
        """
        Return the viewport sub-rectangle outside the operating-system edge reserve, or None when the reserve consumes it.
        """

        reserve_top = int(viewport.height * policy.top)
        reserve_side = int(viewport.width * policy.side)
        reserve_bottom = int(viewport.height * policy.bottom)

        width = viewport.width - (reserve_side * 2)
        height = viewport.height - reserve_top - reserve_bottom

        if width < 1 or height < 1:
            return None

        return Bounds(
            width=width,
            height=height,
            source=viewport.source,
            y=viewport.y + reserve_top,
            x=viewport.x + reserve_side,
            coordinate_system=viewport.system,
        )

    def confine(
        self,
        *,
        path: GesturePath,
        viewport: Bounds,
        policy: ReservePolicy,
    ) -> Optional[GesturePath]:
        """
        Return the path with its touch-down point pulled inside the addressable area, or None when no addressable area exists.
        """

        area = self.addressable(viewport=viewport, policy=policy)
        if area is None:
            return None

        start_x = self.__clamp(value=path.start_x, lower=area.x, upper=area.x + area.width)
        start_y = self.__clamp(value=path.start_y, lower=area.y, upper=area.y + area.height)

        if start_x == path.start_x and start_y == path.start_y:
            return path

        return path.model_copy(update={"start_x": start_x, "start_y": start_y})

    @staticmethod
    def __clamp(*, value: int, lower: int, upper: int) -> int:
        """
        Return the value confined to the inclusive range.
        """

        return max(lower, min(value, upper))
