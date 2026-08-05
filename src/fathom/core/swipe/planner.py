from __future__ import annotations

from typing import List, Optional, Tuple

from fathom.constants.observation import KeyboardVisibility
from fathom.constants.swipe import AbortReason, RetryDirection
from fathom.core.swipe.anchor import AnchorGuard
from fathom.schemas.actions import Bounds, GesturePath
from fathom.schemas.observation import KeyboardObservation
from fathom.schemas.swipe import (
    Admission,
    CandidateSequence,
    ReservePolicy,
    SwipeRejection,
    SwipeRetryPolicy,
)


class SwipeRetryPlanner:
    """
    Pure-logic planner that generates ordered retry candidates with bounds, travel and keyboard filters.
    """

    def __init__(self, *, guard: Optional[AnchorGuard] = None) -> None:
        """
        Bind the guard that excludes operating-system reserved screen edges from touch-down points.
        """

        self.__guard = guard if guard is not None else AnchorGuard()

    def candidates(
        self,
        *,
        original: GesturePath,
        bounds: Bounds,
        policy: SwipeRetryPolicy,
        keyboard: KeyboardObservation,
        frame: Optional[Bounds] = None,
    ) -> CandidateSequence:
        """
        Return ordered accepted candidates (original plus retries) with all safety filters applied uniformly.

        Confinement can collapse distinct shifts onto one anchor, so duplicates
        are dropped rather than dispatched twice against the same coordinates.
        """

        accepted: List[GesturePath] = []
        rejections: List[SwipeRejection] = []
        keyboard_bounds = self.__keyboard_bounds(keyboard=keyboard)

        admission = self.__admit(
            path=original,
            frame=frame,
            bounds=bounds,
            keyboard_bounds=keyboard_bounds,
            minimum_travel=policy.minimum_travel,
            reserve=policy.reserve,
        )
        if admission.reason is None:
            accepted.append(admission.path)
        else:
            rejections.append(
                SwipeRejection(index=0, path=admission.path, reason=admission.reason),
            )

        if not policy.enabled:
            return CandidateSequence(accepted=tuple(accepted), rejections=tuple(rejections))

        for index, magnitude in enumerate(policy.magnitudes, start=1):
            shifted_paths = self.__shifted_paths(
                original=original,
                magnitude=magnitude,
                direction=policy.direction,
            )
            for shifted in shifted_paths:
                shifted_admission = self.__admit(
                    path=shifted,
                    frame=frame,
                    bounds=bounds,
                    reserve=policy.reserve,
                    keyboard_bounds=keyboard_bounds,
                    minimum_travel=policy.minimum_travel,
                )
                if shifted_admission.reason is not None:
                    rejections.append(
                        SwipeRejection(
                            index=index,
                            path=shifted_admission.path,
                            reason=shifted_admission.reason,
                        ),
                    )
                    continue

                if shifted_admission.path in accepted:
                    continue

                accepted.append(shifted_admission.path)

        return CandidateSequence(accepted=tuple(accepted), rejections=tuple(rejections))

    def __shifted_paths(
        self,
        *,
        original: GesturePath,
        magnitude: float,
        direction: RetryDirection,
    ) -> Tuple[GesturePath, ...]:
        """
        Build the candidate paths for one magnitude per the requested retry direction.
        """

        travel = original.distance
        if travel <= 0:
            return ()

        if self.__is_vertical(path=original):
            return self.__shift_axis(
                original=original,
                magnitude=magnitude,
                direction=direction,
                travel=travel,
                axis="y",
            )
        return self.__shift_axis(
            original=original,
            magnitude=magnitude,
            direction=direction,
            travel=travel,
            axis="x",
        )

    @staticmethod
    def __is_vertical(*, path: GesturePath) -> bool:
        """
        Heuristic axis detector based on dominant displacement.
        """

        return abs(path.end_y - path.start_y) >= abs(path.end_x - path.start_x)

    def __shift_axis(
        self,
        *,
        original: GesturePath,
        magnitude: float,
        direction: RetryDirection,
        travel: int,
        axis: str,
    ) -> Tuple[GesturePath, ...]:
        """
        Produce shifted gesture candidates along the dominant axis for the configured direction.
        """

        shift_distance = max(1, int(round(magnitude * travel)))
        inward_sign = self.__inward_sign(original=original, axis=axis)
        signs: List[int]
        if direction is RetryDirection.INWARD:
            signs = [inward_sign]
        elif direction is RetryDirection.OUTWARD:
            signs = [-inward_sign]
        else:
            signs = [inward_sign, -inward_sign]

        paths: List[GesturePath] = []
        for sign in signs:
            shifted = self.__apply_shift(
                original=original,
                axis=axis,
                delta=sign * shift_distance,
            )
            paths.append(shifted)
        return tuple(paths)

    @staticmethod
    def __inward_sign(*, original: GesturePath, axis: str) -> int:
        """
        Return the sign that moves the start toward the end on the dominant axis.
        """

        if axis == "y":
            return 1 if original.end_y > original.start_y else -1
        return 1 if original.end_x > original.start_x else -1

    @staticmethod
    def __apply_shift(*, original: GesturePath, axis: str, delta: int) -> GesturePath:
        """
        Translate only the start coordinate along the given axis; end and duration unchanged.
        """

        if axis == "y":
            return original.model_copy(update={"start_y": max(0, original.start_y + delta)})
        return original.model_copy(update={"start_x": max(0, original.start_x + delta)})

    def __admit(
        self,
        *,
        bounds: Bounds,
        path: GesturePath,
        minimum_travel: int,
        reserve: ReservePolicy,
        frame: Optional[Bounds] = None,
        keyboard_bounds: Optional[Bounds],
    ) -> Admission:
        """
        Confine the candidate's touch-down to addressable screen area and return its dispatch verdict.
        """

        confined = self.__guard.confine(path=path, viewport=bounds, policy=reserve)

        if confined is None:
            return Admission(path=path, reason=AbortReason.ANCHOR_RESERVED)

        return Admission(
            path=confined,
            reason=self.__reject(
                frame=frame,
                bounds=bounds,
                path=confined,
                minimum_travel=minimum_travel,
                keyboard_bounds=keyboard_bounds,
            ),
        )

    @staticmethod
    def __reject(
        *,
        bounds: Bounds,
        path: GesturePath,
        minimum_travel: int,
        frame: Optional[Bounds] = None,
        keyboard_bounds: Optional[Bounds],
    ) -> Optional[AbortReason]:
        """
        Return the first applicable rejection reason for the candidate path, or None if it passes.

        Only a path that exits the screen rejects here; reserved edges are confined in __admit,
        and travel and keyboard concerns are left to post-dispatch observation and retry.
        """

        _ = keyboard_bounds, minimum_travel

        if not SwipeRetryPlanner.__inside_bounds(path=path, bounds=bounds):
            return AbortReason.OUT_OF_BOUNDS

        if frame is not None and not SwipeRetryPlanner.__anchor_inside(path=path, frame=frame):
            return AbortReason.UNSAFE_ANCHOR

        return None

    @staticmethod
    def __anchor_inside(*, path: GesturePath, frame: Bounds) -> bool:
        """
        Whether the touch-down anchor lies inside the OS-reported window frame; the endpoint may exit freely.
        """

        return (
            frame.x <= path.start_x <= frame.x + frame.width
            and frame.y <= path.start_y <= frame.y + frame.height
        )

    @staticmethod
    def __keyboard_bounds(*, keyboard: KeyboardObservation) -> Optional[Bounds]:
        """
        Return the keyboard bounds only when visibility is VISIBLE and bounds are known.
        """

        if keyboard.visibility is KeyboardVisibility.VISIBLE and keyboard.bounds is not None:
            return keyboard.bounds
        return None

    @staticmethod
    def __inside_bounds(*, path: GesturePath, bounds: Bounds) -> bool:
        """
        Whether both endpoints lie inside the supplied rectangle.
        """

        x_left, x_right = bounds.x, bounds.x + bounds.width
        y_top, y_bottom = bounds.y, bounds.y + bounds.height

        return (
            x_left <= path.start_x <= x_right
            and x_left <= path.end_x <= x_right
            and y_top <= path.start_y <= y_bottom
            and y_top <= path.end_y <= y_bottom
        )

    @staticmethod
    def __travel(*, path: GesturePath) -> int:
        """
        Return the axis-dominant gesture travel distance in pixels.
        """

        delta_y = abs(path.end_y - path.start_y)
        delta_x = abs(path.end_x - path.start_x)
        return delta_y if delta_y >= delta_x else delta_x

    @staticmethod
    def __intersects(*, path: GesturePath, region: Bounds) -> bool:
        """
        Whether the line segment from start to end intersects the rectangle.
        """

        return SwipeRetryPlanner.__segment_rect_intersect(
            x1=path.start_x,
            y1=path.start_y,
            x2=path.end_x,
            y2=path.end_y,
            rx=region.x,
            ry=region.y,
            rw=region.width,
            rh=region.height,
        )

    @staticmethod
    def __segment_rect_intersect(
        *,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        rx: int,
        ry: int,
        rw: int,
        rh: int,
    ) -> bool:
        """
        Liang-Barsky line-rectangle intersection test in integer pixel space.
        """

        if SwipeRetryPlanner.__point_in_rect(x=x1, y=y1, rx=rx, ry=ry, rw=rw, rh=rh):
            return True

        if SwipeRetryPlanner.__point_in_rect(x=x2, y=y2, rx=rx, ry=ry, rw=rw, rh=rh):
            return True

        x_right, y_bottom = rx + rw, ry + rh

        for edge in (
            (rx, ry, x_right, ry),
            (rx, ry, rx, y_bottom),
            (x_right, ry, x_right, y_bottom),
            (rx, y_bottom, x_right, y_bottom),
        ):
            if SwipeRetryPlanner.__segments_intersect(
                a1x=x1,
                a1y=y1,
                a2x=x2,
                a2y=y2,
                b1x=edge[0],
                b1y=edge[1],
                b2x=edge[2],
                b2y=edge[3],
            ):
                return True
        return False

    @staticmethod
    def __point_in_rect(*, x: int, y: int, rx: int, ry: int, rw: int, rh: int) -> bool:
        """
        Whether a point lies inside the rectangle (inclusive of edges).
        """

        return rx <= x <= rx + rw and ry <= y <= ry + rh

    @staticmethod
    def __segments_intersect(
        *,
        a1x: int,
        a1y: int,
        a2x: int,
        a2y: int,
        b1x: int,
        b1y: int,
        b2x: int,
        b2y: int,
    ) -> bool:
        """
        Whether two line segments share a point, using orientation tests.
        """

        def orientation(px: int, py: int, qx: int, qy: int, rx: int, ry: int) -> int:
            value = (qy - py) * (rx - qx) - (qx - px) * (ry - qy)
            if value > 0:
                return 1
            if value < 0:
                return -1
            return 0

        o1 = orientation(a1x, a1y, a2x, a2y, b1x, b1y)
        o2 = orientation(a1x, a1y, a2x, a2y, b2x, b2y)
        o3 = orientation(b1x, b1y, b2x, b2y, a1x, a1y)
        o4 = orientation(b1x, b1y, b2x, b2y, a2x, a2y)
        return o1 != o2 and o3 != o4
