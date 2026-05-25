from __future__ import annotations

import unittest

from fathom.constants.observation import KeyboardVisibility
from fathom.constants.swipe import AbortReason, RetryDirection
from fathom.core.swipe.planner import SwipeRetryPlanner
from fathom.schemas.actions import Bounds, CoordinateSystem, GesturePath
from fathom.schemas.observation import KeyboardObservation
from fathom.schemas.swipe import SwipeRetryPolicy


class SwipeRetryPlannerVerticalTest(unittest.TestCase):
    """
    Verify vertical-swipe retry planning under bounds, minimum-travel, and keyboard filters.
    """

    @staticmethod
    def __viewport_bounds() -> Bounds:
        """
        Build a 1080x2208 device-pixel viewport rectangle.
        """

        return Bounds(
            x=0,
            y=0,
            width=1080,
            height=2208,
            coordinate_system=CoordinateSystem.DEVICE_PIXEL,
        )

    @staticmethod
    def __upward_path(start_y: int = 2008, end_y: int = 800) -> GesturePath:
        """
        Build an upward swipe path centered on the device.
        """

        return GesturePath(start_x=540, start_y=start_y, end_x=540, end_y=end_y, duration=300)

    def test_keyboard_blocked_returns_only_rejection(self) -> None:
        """
        Original gesture intersecting the visible keyboard is rejected for KEYBOARD_BLOCKED.
        """

        keyboard = KeyboardObservation(
            visibility=KeyboardVisibility.VISIBLE,
            bounds=Bounds(
                x=0,
                y=1507,
                width=1080,
                height=701,
                coordinate_system=CoordinateSystem.DEVICE_PIXEL,
            ),
        )
        policy = SwipeRetryPolicy(enabled=False)

        sequence = SwipeRetryPlanner().candidates(
            original=self.__upward_path(),
            bounds=self.__viewport_bounds(),
            policy=policy,
            keyboard=keyboard,
        )

        self.assertEqual(len(sequence.accepted), 0)
        self.assertEqual(len(sequence.rejections), 1)
        self.assertEqual(sequence.rejections[0].reason, AbortReason.KEYBOARD_BLOCKED)

    def test_original_accepted_when_no_keyboard(self) -> None:
        """
        With visibility UNKNOWN and retry disabled, only the original gesture is accepted.
        """

        policy = SwipeRetryPolicy(enabled=False)

        sequence = SwipeRetryPlanner().candidates(
            original=self.__upward_path(),
            bounds=self.__viewport_bounds(),
            policy=policy,
            keyboard=KeyboardObservation(visibility=KeyboardVisibility.UNKNOWN),
        )

        self.assertEqual(len(sequence.accepted), 1)
        self.assertEqual(len(sequence.rejections), 0)

    def test_inward_retry_shifts_origin_toward_endpoint(self) -> None:
        """
        Three INWARD retries shift the start downward (toward the end) for an upward gesture.
        """

        policy = SwipeRetryPolicy(
            enabled=True,
            direction=RetryDirection.INWARD,
            magnitudes=(0.10, 0.20, 0.30),
        )
        original = self.__upward_path(start_y=2000, end_y=400)

        sequence = SwipeRetryPlanner().candidates(
            original=original,
            bounds=self.__viewport_bounds(),
            policy=policy,
            keyboard=KeyboardObservation(visibility=KeyboardVisibility.HIDDEN),
        )

        starts = [path.start_y for path in sequence.accepted]
        self.assertEqual(len(starts), 4)
        self.assertEqual(starts[0], 2000)
        self.assertLess(starts[1], starts[0])
        self.assertLess(starts[2], starts[1])
        self.assertLess(starts[3], starts[2])

    def test_retries_blocked_by_keyboard_are_rejected(self) -> None:
        """
        Retry candidates whose shifted start lands inside the keyboard region are rejected.
        """

        policy = SwipeRetryPolicy(
            enabled=True,
            direction=RetryDirection.OUTWARD,
            magnitudes=(0.10,),
        )
        keyboard = KeyboardObservation(
            visibility=KeyboardVisibility.VISIBLE,
            bounds=Bounds(
                x=0,
                y=1900,
                width=1080,
                height=308,
                coordinate_system=CoordinateSystem.DEVICE_PIXEL,
            ),
        )

        sequence = SwipeRetryPlanner().candidates(
            original=self.__upward_path(start_y=1850, end_y=400),
            bounds=self.__viewport_bounds(),
            policy=policy,
            keyboard=keyboard,
        )

        rejection_reasons = {rejection.reason for rejection in sequence.rejections}
        self.assertIn(AbortReason.KEYBOARD_BLOCKED, rejection_reasons)

    def test_minimum_travel_violation_rejected(self) -> None:
        """
        Candidates that collapse below ``minimum_travel`` are rejected for MINIMUM_TRAVEL_VIOLATED.
        """

        policy = SwipeRetryPolicy(enabled=False, minimum_travel=300)
        sequence = SwipeRetryPlanner().candidates(
            original=GesturePath(start_x=540, start_y=900, end_x=540, end_y=850, duration=300),
            bounds=self.__viewport_bounds(),
            policy=policy,
            keyboard=KeyboardObservation(visibility=KeyboardVisibility.HIDDEN),
        )

        self.assertEqual(len(sequence.accepted), 0)
        self.assertEqual(sequence.rejections[0].reason, AbortReason.MINIMUM_TRAVEL_VIOLATED)


class SwipeRetryPlannerHorizontalTest(unittest.TestCase):
    """
    Cover horizontal swipe planning where the dominant axis is X.
    """

    def test_horizontal_inward_shift_moves_start_along_x(self) -> None:
        """
        For a left-to-right gesture, INWARD retries move the start rightward (toward the end).
        """

        planner = SwipeRetryPlanner()
        original = GesturePath(start_x=200, start_y=1100, end_x=900, end_y=1100, duration=300)
        bounds = Bounds(
            x=0,
            y=0,
            width=1080,
            height=2208,
            coordinate_system=CoordinateSystem.DEVICE_PIXEL,
        )

        sequence = planner.candidates(
            original=original,
            bounds=bounds,
            policy=SwipeRetryPolicy(
                enabled=True,
                direction=RetryDirection.INWARD,
                magnitudes=(0.10, 0.20),
            ),
            keyboard=KeyboardObservation(visibility=KeyboardVisibility.HIDDEN),
        )

        starts = [path.start_x for path in sequence.accepted]
        self.assertEqual(starts[0], 200)
        self.assertGreater(starts[1], starts[0])
        self.assertGreater(starts[2], starts[1])
