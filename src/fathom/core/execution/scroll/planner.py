from __future__ import annotations

import json
from logging import getLogger
from typing import Optional

from fathom.constants.scroll import (
    DEFAULT_SCROLL_MINIMUM_DISTANCE,
    DEFAULT_SCROLL_SHIFT_PRIMARY_RATIO,
    DEFAULT_SCROLL_SHIFT_SECONDARY_RATIO,
    ScrollStage,
    ScrollVerdictKind,
)
from fathom.interfaces.scroll import ScrollPlanPort
from fathom.schemas.actions import CoordinateSource, ExecutionRegion, GesturePath
from fathom.schemas.configuration import ScrollInteractionPolicy
from fathom.schemas.scroll import ScrollAttempt, ScrollContext, ScrollScope, ScrollSurface
from fathom.utils.coordinates import CoordinateConverter

logger = getLogger(__name__)


class ScrollPlanner(ScrollPlanPort):
    """
    Plans bounded scroll attempts inside one resolved scope while preserving the first gesture.
    """

    def plan(
        self,
        *,
        context: ScrollContext,
        current: GesturePath,
        scope: Optional[ScrollScope],
        surfaces: tuple[ScrollSurface, ...] = (),
        converter: CoordinateConverter,
        policy: ScrollInteractionPolicy.AdaptivePolicy,
        capture_height: int,
    ) -> tuple[ScrollAttempt, ...]:
        """
        Build ordered scope-constrained attempts.
        """

        _ = context

        if scope is None:
            raise ValueError("ScrollPlanner requires a resolved scroll scope.")

        base_path = self.__refine_path(
            current=self.__initial_path(current=current, scope=scope, converter=converter),
            scope=scope,
            converter=converter,
            surfaces=surfaces,
            policy=policy,
            capture_height=capture_height,
        )
        attempts: list[ScrollAttempt] = []

        current_attempt = self.__build_attempt(
            stage=ScrollStage.CURRENT,
            path=base_path,
            region=scope.region,
            scope=scope,
            avoided=surfaces,
        )
        if current_attempt is not None:
            attempts.append(current_attempt)

        for index, ratio in enumerate(
            (DEFAULT_SCROLL_SHIFT_PRIMARY_RATIO, DEFAULT_SCROLL_SHIFT_SECONDARY_RATIO),
            start=1,
        ):
            shifted_path = self.__refine_path(
                current=self.__shift_path(
                    path=base_path,
                    scope=scope,
                    converter=converter,
                    delta_ratio=ratio,
                    sign=1 if index == 1 else -1,
                ),
                scope=scope,
                converter=converter,
                surfaces=surfaces,
                policy=policy,
                capture_height=capture_height,
            )
            shifted_attempt = self.__build_attempt(
                stage=ScrollStage.SHIFT if index == 1 else ScrollStage.SHORT,
                path=shifted_path,
                region=scope.region,
                scope=scope,
                avoided=surfaces,
            )
            if shifted_attempt is None or self.__duplicate(
                path=shifted_attempt.path,
                attempts=tuple(attempts),
            ):
                continue
            attempts.append(shifted_attempt)

        return tuple(attempts)

    def __refine_path(
        self,
        *,
        current: GesturePath,
        scope: ScrollScope,
        converter: CoordinateConverter,
        surfaces: tuple[ScrollSurface, ...],
        policy: ScrollInteractionPolicy.AdaptivePolicy,
        capture_height: int,
    ) -> GesturePath:
        """
        Clamp the gesture into scope and move it away from interfering surface bands.
        """

        clamped = self.__clamp_path_to_scope(path=current, scope=scope, converter=converter)
        adjusted, detail = self.__avoid_surfaces(
            path=clamped,
            scope=scope,
            surfaces=surfaces,
            policy=policy,
            capture_height=capture_height,
        )
        self.__log_refinement(
            scope=scope,
            original=current,
            clamped=clamped,
            adjusted=adjusted,
            detail=detail,
        )
        if self.__has_meaningful_travel(path=adjusted):
            return adjusted
        return clamped

    def log_attempt(self, *, attempt: ScrollAttempt, attempt_index: int) -> None:
        """
        Log one dispatched scope-constrained attempt.
        """

        attempt_path = attempt.path.model_dump()
        attempt_path["distance"] = attempt.path.distance
        logger.info(
            json.dumps(
                {
                    "component": "core.execution.scroll",
                    "event": "scroll.attempt.dispatch",
                    "attempt.index": attempt_index,
                    "attempt.stage": int(attempt.stage),
                    "scope.identifier": attempt.scope.identifier,
                    "scope.kind": attempt.scope.kind.value,
                    "path": attempt_path,
                    "capture_region": attempt.capture_region.model_dump(),
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )

    def ambiguous_verdict_kind(self) -> ScrollVerdictKind:
        """
        Return the verdict kind representing an ambiguous outcome.
        """

        return ScrollVerdictKind.AMBIGUOUS

    def __initial_path(
        self,
        *,
        current: GesturePath,
        scope: ScrollScope,
        converter: CoordinateConverter,
    ) -> GesturePath:
        """
        Preserve the original gesture when it remains valid inside the scope.
        """

        clamped = self.__clamp_path_to_scope(path=current, scope=scope, converter=converter)
        if self.__has_meaningful_travel(path=clamped):
            return clamped

        return self.__fallback_path(scope=scope, current=current, converter=converter)

    def __fallback_path(
        self,
        *,
        scope: ScrollScope,
        current: GesturePath,
        converter: CoordinateConverter,
    ) -> GesturePath:
        """
        Rebuild one centered path only when the original gesture becomes unusable after clamping.
        """

        region = converter.region_from_bounds(
            bounds=scope.bounds,
            source=scope.bounds.source or CoordinateSource.VIEWPORT,
        )
        if self.__is_vertical(path=current):
            return converter.resolve_scroll_path(
                region=region,
                direction="down" if current.start_y > current.end_y else "up",
            )

        return converter.resolve_swipe_path(
            region=region,
            direction="left" if current.start_x > current.end_x else "right",
        )

    def __shift_path(
        self,
        *,
        path: GesturePath,
        scope: ScrollScope,
        converter: CoordinateConverter,
        delta_ratio: float,
        sign: int,
    ) -> GesturePath:
        """
        Shift the path within the same container by a small bounded delta.
        """

        clamped = self.__clamp_path_to_scope(path=path, scope=scope, converter=converter)
        if self.__is_vertical(path=clamped):
            delta = max(1, int(scope.region.width * delta_ratio))
            shifted = GesturePath(
                start_x=clamped.start_x + (delta * sign),
                start_y=clamped.start_y,
                end_x=clamped.end_x + (delta * sign),
                end_y=clamped.end_y,
                duration=clamped.duration,
            )
            return self.__clamp_path_to_scope(path=shifted, scope=scope, converter=converter)

        delta = max(1, int(scope.region.height * delta_ratio))
        shifted = GesturePath(
            start_x=clamped.start_x,
            start_y=clamped.start_y + (delta * sign),
            end_x=clamped.end_x,
            end_y=clamped.end_y + (delta * sign),
            duration=clamped.duration,
        )
        return self.__clamp_path_to_scope(path=shifted, scope=scope, converter=converter)

    def __clamp_path_to_scope(
        self,
        *,
        path: GesturePath,
        scope: ScrollScope,
        converter: CoordinateConverter,
    ) -> GesturePath:
        """
        Clamp one raw path into the resolved scope.
        """

        region = converter.region_from_bounds(
            bounds=scope.bounds,
            source=scope.bounds.source or CoordinateSource.VIEWPORT,
        )
        start_x = max(region.x, min(path.start_x, region.x + region.width - 1))
        end_x = max(region.x, min(path.end_x, region.x + region.width - 1))
        start_y = max(region.y, min(path.start_y, region.y + region.height - 1))
        end_y = max(region.y, min(path.end_y, region.y + region.height - 1))
        return GesturePath(
            start_x=start_x,
            start_y=start_y,
            end_x=end_x,
            end_y=end_y,
            duration=path.duration,
        )

    def __build_attempt(
        self,
        *,
        stage: ScrollStage,
        path: GesturePath,
        region: ExecutionRegion,
        scope: ScrollScope,
        avoided: tuple[ScrollSurface, ...],
    ) -> Optional[ScrollAttempt]:
        """
        Build one valid attempt inside the resolved scope.
        """

        if path.distance < DEFAULT_SCROLL_MINIMUM_DISTANCE:
            return None

        return ScrollAttempt(
            stage=stage,
            path=path,
            region=region,
            scope=scope,
            capture_region=scope.bounds,
            avoided=avoided,
        )

    def __avoid_surfaces(
        self,
        *,
        path: GesturePath,
        scope: ScrollScope,
        surfaces: tuple[ScrollSurface, ...],
        policy: ScrollInteractionPolicy.AdaptivePolicy,
        capture_height: int,
    ) -> tuple[GesturePath, dict[str, object]]:
        """
        Move the gesture away from sticky/footer bands when they intersect the active lane.
        """

        if not surfaces:
            return path, {
                "strategy": "none",
                "changed": False,
                "reason": "no_surfaces",
                "surface.count": 0,
            }

        if self.__is_vertical(path=path):
            return self.__avoid_vertical_surfaces(
                path=path,
                scope=scope,
                surfaces=surfaces,
                policy=policy,
                capture_height=capture_height,
            )

        return self.__avoid_horizontal_surfaces(
            path=path,
            scope=scope,
            surfaces=surfaces,
            policy=policy,
            capture_height=capture_height,
        )

    def __avoid_vertical_surfaces(
        self,
        *,
        path: GesturePath,
        scope: ScrollScope,
        surfaces: tuple[ScrollSurface, ...],
        policy: ScrollInteractionPolicy.AdaptivePolicy,
        capture_height: int,
    ) -> tuple[GesturePath, dict[str, object]]:
        """
        Adapt vertical anchors using geometry, not semantic surface labels.
        """

        region_top = scope.region.y
        region_bottom = scope.region.y + scope.region.height - 1
        lane_left = min(path.start_x, path.end_x)
        lane_right = max(path.start_x, path.end_x)
        suspicious_start = region_top + int(scope.region.height * policy.suspicious_bottom_ratio)
        start_cap = min(region_bottom, suspicious_start)
        end_floor = region_top
        affected_surfaces: list[dict[str, object]] = []

        for surface in surfaces:
            if not self.__overlaps_horizontally(
                lane_left=lane_left,
                lane_right=lane_right,
                surface=surface,
            ):
                continue

            surface_top = surface.bounds.y
            surface_bottom = surface.bounds.y + surface.bounds.height - 1
            if surface_top <= path.start_y <= surface_bottom:
                start_cap = min(start_cap, surface_top - 1)
                affected_surfaces.append(self.__surface_detail(surface=surface, effect="start_cap"))
            if surface_top <= path.end_y <= surface_bottom:
                end_floor = max(end_floor, surface_bottom + 1)
                affected_surfaces.append(self.__surface_detail(surface=surface, effect="end_floor"))

        start_y = path.start_y
        end_y = path.end_y

        if start_y > start_cap:
            delta = start_y - start_cap
            start_y -= delta
            end_y -= delta

        if end_y < end_floor:
            delta = end_floor - end_y
            start_y += delta
            end_y += delta

        start_y = min(start_y, region_bottom)
        end_y = max(end_y, region_top)

        if start_y > start_cap:
            start_y = start_cap
        if end_y < end_floor:
            end_y = end_floor

        if start_y - end_y < DEFAULT_SCROLL_MINIMUM_DISTANCE:
            return path, {
                "strategy": "vertical_anchor_adjustment",
                "changed": False,
                "rejected": True,
                "reason": "insufficient_travel_after_adjustment",
                "surface.count": len(surfaces),
                "surfaces.affected": affected_surfaces,
                "suspicious.start": suspicious_start,
                "start.cap": start_cap,
                "end.floor": end_floor,
            }

        adjusted = GesturePath(
            start_x=path.start_x,
            start_y=start_y,
            end_x=path.end_x,
            end_y=end_y,
            duration=path.duration,
        )
        return adjusted, {
            "strategy": "vertical_anchor_adjustment",
            "changed": adjusted != path,
            "rejected": False,
            "reason": "unsafe_vertical_anchor_detected",
            "surface.count": len(surfaces),
            "surfaces.affected": affected_surfaces,
            "suspicious.start": suspicious_start,
            "start.cap": start_cap,
            "end.floor": end_floor,
        }

    def __avoid_horizontal_surfaces(
        self,
        *,
        path: GesturePath,
        scope: ScrollScope,
        surfaces: tuple[ScrollSurface, ...],
        policy: ScrollInteractionPolicy.AdaptivePolicy,
        capture_height: int,
    ) -> tuple[GesturePath, dict[str, object]]:
        """
        Shift the horizontal lane away from interfering vertical bands geometrically.
        """

        _ = capture_height

        region_top = scope.region.y
        region_bottom = scope.region.y + scope.region.height - 1
        lane_top = min(path.start_y, path.end_y)
        lane_bottom = max(path.start_y, path.end_y)
        suspicious_band_top = region_top + int(scope.region.height * policy.suspicious_bottom_ratio)
        preferred_y = min(path.start_y, suspicious_band_top)
        upper_limit = region_top
        lower_limit = region_bottom
        affected_surfaces: list[dict[str, object]] = []

        for surface in surfaces:
            if not self.__overlaps_vertically(
                lane_top=lane_top,
                lane_bottom=lane_bottom,
                surface=surface,
            ):
                continue

            surface_top = surface.bounds.y
            surface_bottom = surface.bounds.y + surface.bounds.height - 1
            if surface_top <= preferred_y <= surface_bottom:
                if preferred_y >= (region_top + scope.region.height / 2):
                    lower_limit = min(lower_limit, surface_top - 1)
                    affected_surfaces.append(
                        self.__surface_detail(surface=surface, effect="lower_limit")
                    )
                else:
                    upper_limit = max(upper_limit, surface_bottom + 1)
                    affected_surfaces.append(
                        self.__surface_detail(surface=surface, effect="upper_limit")
                    )
            elif surface_top >= preferred_y:
                lower_limit = min(lower_limit, surface_top - 1)
            else:
                upper_limit = max(upper_limit, surface_bottom + 1)

        mid_y = min(max(preferred_y, upper_limit), lower_limit)
        if not (region_top <= mid_y <= region_bottom):
            return path, {
                "strategy": "horizontal_lane_adjustment",
                "changed": False,
                "rejected": True,
                "reason": "adjusted_lane_out_of_scope",
                "surface.count": len(surfaces),
                "surfaces.affected": affected_surfaces,
                "preferred.y": preferred_y,
                "upper.limit": upper_limit,
                "lower.limit": lower_limit,
            }

        adjusted = GesturePath(
            start_x=path.start_x,
            start_y=mid_y,
            end_x=path.end_x,
            end_y=mid_y,
            duration=path.duration,
        )
        return adjusted, {
            "strategy": "horizontal_lane_adjustment",
            "changed": adjusted != path,
            "rejected": False,
            "reason": "unsafe_horizontal_lane_detected",
            "surface.count": len(surfaces),
            "surfaces.affected": affected_surfaces,
            "preferred.y": preferred_y,
            "upper.limit": upper_limit,
            "lower.limit": lower_limit,
        }

    def __log_refinement(
        self,
        *,
        scope: ScrollScope,
        original: GesturePath,
        clamped: GesturePath,
        adjusted: GesturePath,
        detail: dict[str, object],
    ) -> None:
        """
        Emit one structured record describing the adaptive geometry decision.
        """

        logger.info(
            json.dumps(
                {
                    "component": "core.execution.scroll",
                    "event": "scroll.path.refined",
                    "scope.identifier": scope.identifier,
                    "scope.kind": scope.kind.value,
                    "scope.axis": scope.axis,
                    "strategy": detail.get("strategy"),
                    "reason": detail.get("reason"),
                    "changed": bool(detail.get("changed", False)),
                    "rejected": bool(detail.get("rejected", False)),
                    "surface.count": detail.get("surface.count", 0),
                    "surfaces.affected": detail.get("surfaces.affected", ()),
                    "suspicious.start": detail.get("suspicious.start"),
                    "start.cap": detail.get("start.cap"),
                    "end.floor": detail.get("end.floor"),
                    "preferred.y": detail.get("preferred.y"),
                    "upper.limit": detail.get("upper.limit"),
                    "lower.limit": detail.get("lower.limit"),
                    "original.path": self.__path_payload(path=original),
                    "clamped.path": self.__path_payload(path=clamped),
                    "adjusted.path": self.__path_payload(path=adjusted),
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )

    @staticmethod
    def __surface_detail(*, surface: ScrollSurface, effect: str) -> dict[str, object]:
        """
        Build one compact structured surface detail for logging.
        """

        return {
            "kind": surface.kind.value,
            "effect": effect,
            "detail": surface.detail,
            "bounds": {
                "x": surface.bounds.x,
                "y": surface.bounds.y,
                "width": surface.bounds.width,
                "height": surface.bounds.height,
            },
        }

    @staticmethod
    def __path_payload(*, path: GesturePath) -> dict[str, int]:
        """
        Return one stable log payload for a gesture path.
        """

        return {
            "start_x": path.start_x,
            "start_y": path.start_y,
            "end_x": path.end_x,
            "end_y": path.end_y,
            "duration": path.duration,
            "distance": path.distance,
        }

    @staticmethod
    def __overlaps_horizontally(
        *,
        lane_left: int,
        lane_right: int,
        surface: ScrollSurface,
    ) -> bool:
        """
        Return whether the active vertical lane intersects one surface horizontally.
        """

        surface_left = surface.bounds.x
        surface_right = surface.bounds.x + surface.bounds.width - 1
        return not (lane_right < surface_left or lane_left > surface_right)

    @staticmethod
    def __overlaps_vertically(
        *,
        lane_top: int,
        lane_bottom: int,
        surface: ScrollSurface,
    ) -> bool:
        """
        Return whether the active horizontal lane intersects one surface vertically.
        """

        surface_top = surface.bounds.y
        surface_bottom = surface.bounds.y + surface.bounds.height - 1
        return not (lane_bottom < surface_top or lane_top > surface_bottom)

    @staticmethod
    def __duplicate(*, path: GesturePath, attempts: tuple[ScrollAttempt, ...]) -> bool:
        """
        Return whether the proposed path duplicates an earlier attempt.
        """

        return any(
            attempt.path.start_x == path.start_x
            and attempt.path.start_y == path.start_y
            and attempt.path.end_x == path.end_x
            and attempt.path.end_y == path.end_y
            for attempt in attempts
        )

    @staticmethod
    def __has_meaningful_travel(*, path: GesturePath) -> bool:
        """
        Return whether the path still carries useful travel after clamping.
        """

        return path.distance >= DEFAULT_SCROLL_MINIMUM_DISTANCE

    @staticmethod
    def __is_vertical(*, path: GesturePath) -> bool:
        """
        Return whether the gesture is primarily vertical.
        """

        return abs(path.start_y - path.end_y) >= abs(path.start_x - path.end_x)
