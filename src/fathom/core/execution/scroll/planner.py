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
        _ = policy
        _ = capture_height

        if scope is None:
            raise ValueError("ScrollPlanner requires a resolved scroll scope.")

        base_path = self.__initial_path(current=current, scope=scope, converter=converter)
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
            shifted_path = self.__shift_path(
                path=base_path,
                scope=scope,
                converter=converter,
                delta_ratio=ratio,
                sign=1 if index == 1 else -1,
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
