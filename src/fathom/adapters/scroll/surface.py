from __future__ import annotations

from typing import Tuple

from fathom.constants.scroll import ScrollEvidenceSource, SurfaceKind
from fathom.interfaces.scroll import ScrollSurfacePort
from fathom.schemas.actions import Bounds, CoordinateSystem, GesturePath
from fathom.schemas.observation import ElementRole, ScreenObservation
from fathom.schemas.scroll import ScrollSurface


class ScrollSurfaceInspector(ScrollSurfacePort):
    """
    Extracts unsafe surface hints from the existing screen observation.
    """

    async def inspect(
        self,
        *,
        observation: ScreenObservation,
        path: GesturePath,
        capture_width: int,
        capture_height: int,
    ) -> Tuple[ScrollSurface, ...]:
        """
        Return interfering surface hints for the proposed path.
        """

        hints = []

        if observation.keyboard.visible and observation.keyboard.bounds is not None:
            hints.append(
                ScrollSurface(
                    kind=SurfaceKind.KEYBOARD,
                    bounds=observation.keyboard.bounds,
                    source=ScrollEvidenceSource.SURFACE,
                    detail="keyboard_visible",
                )
            )

        for overlay in observation.overlays:
            if overlay.visible:
                hints.append(
                    ScrollSurface(
                        bounds=overlay.bounds,
                        detail="blocking_overlay",
                        kind=SurfaceKind.OVERLAY,
                        source=ScrollEvidenceSource.SURFACE,
                    )
                )

        if footer := self.__footer_hint(
            observation=observation,
            capture_height=capture_height,
        ):
            hints.append(footer)

        return tuple(self.__dedupe(hints=tuple(hints)))

    def __footer_hint(
        self,
        *,
        observation: ScreenObservation,
        capture_height: int,
    ) -> ScrollSurface | None:
        """
        Collapse one bottom navigation cluster into a footer surface hint.
        """

        candidates = tuple(
            element
            for element in observation.elements
            if element.tappable
            and element.role is ElementRole.BUTTON
            and element.bounds.y >= int(capture_height * 0.75)
        )

        if len(candidates) < 2:
            return None

        top = min(element.bounds.y for element in candidates)
        bottom = max(element.bounds.y + element.bounds.height for element in candidates)
        left = min(element.bounds.x for element in candidates)
        right = max(element.bounds.x + element.bounds.width for element in candidates)

        return ScrollSurface(
            kind=SurfaceKind.FOOTER,
            bounds=Bounds(
                x=left,
                y=top,
                width=right - left,
                height=bottom - top,
                coordinate_system=CoordinateSystem.DEVICE_PIXEL,
            ),
            source=ScrollEvidenceSource.SURFACE,
            detail="bottom_navigation_cluster",
        )

    def __dedupe(self, *, hints: Tuple[ScrollSurface, ...]) -> Tuple[ScrollSurface, ...]:
        """
        Collapse exact-duplicate surface hints.
        """

        values = {}

        for hint in hints:
            key = (
                hint.kind.value,
                hint.bounds.x,
                hint.bounds.y,
                hint.bounds.width,
                hint.bounds.height,
            )
            values[key] = hint

        return tuple(values.values())
