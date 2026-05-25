from __future__ import annotations

import unittest
from typing import List

from fathom.constants.observation import KeyboardVisibility
from fathom.constants.swipe import AbortReason, RetryDirection
from fathom.core.swipe.coordinator import SwipeRetryCoordinator
from fathom.core.swipe.planner import SwipeRetryPlanner
from fathom.interfaces.swipe import SwipeAttemptDispatcher
from fathom.schemas.actions import Bounds, CoordinateSystem, GesturePath
from fathom.schemas.observation import KeyboardObservation
from fathom.schemas.swipe import (
    DeviceOutcome,
    SwipeAttempt,
    SwipeRetryPolicy,
    VisualOutcome,
)


class _RecordingDispatcher(SwipeAttemptDispatcher):
    """
    Test double that records each dispatched gesture and returns scripted outcomes.
    """

    def __init__(self, *, outcomes: List[SwipeAttempt]) -> None:
        """
        Bind the dispatcher to a fixed sequence of attempt outcomes keyed by call index.
        """

        self.__outcomes = outcomes
        self.received: List[GesturePath] = []

    async def attempt(
        self,
        *,
        path: GesturePath,
        index: int,
        original_before: str,
    ) -> SwipeAttempt:
        """
        Record the dispatched path and return the scripted attempt at ``index``.
        """

        _ = original_before
        self.received.append(path)
        return self.__outcomes[index].model_copy(update={"index": index, "path": path})


class SwipeRetryCoordinatorTest(unittest.IsolatedAsyncioTestCase):
    """
    Exercise the bounded retry orchestration and abort-reason precedence.
    """

    @staticmethod
    def __upward_path() -> GesturePath:
        """
        Build an upward swipe gesture in screen-pixel space.
        """

        return GesturePath(start_x=540, start_y=1900, end_x=540, end_y=600, duration=300)

    @staticmethod
    def __viewport() -> Bounds:
        """
        Build a 1080x2208 device-pixel viewport bounds.
        """

        return Bounds(
            x=0,
            y=0,
            width=1080,
            height=2208,
            coordinate_system=CoordinateSystem.DEVICE_PIXEL,
        )

    async def test_succeeds_on_first_attempt(self) -> None:
        """
        A first attempt with visual change short-circuits subsequent retries.
        """

        ok = SwipeAttempt(
            index=0,
            path=self.__upward_path(),
            device=DeviceOutcome(succeeded=True),
            visual=VisualOutcome(changed=True, before="A", after="B"),
        )
        dispatcher = _RecordingDispatcher(outcomes=[ok])
        coordinator = SwipeRetryCoordinator(planner=SwipeRetryPlanner(), dispatcher=dispatcher)

        execution = await coordinator.execute(
            original=self.__upward_path(),
            bounds=self.__viewport(),
            policy=SwipeRetryPolicy(enabled=True, magnitudes=(0.10, 0.20, 0.30)),
            keyboard=KeyboardObservation(visibility=KeyboardVisibility.HIDDEN),
            original_before="A",
        )

        self.assertEqual(len(execution.attempts), 1)
        self.assertTrue(execution.effective)
        self.assertIsNone(execution.aborted_for)
        self.assertEqual(len(dispatcher.received), 1)

    async def test_aborts_keyboard_blocked_without_any_dispatch(self) -> None:
        """
        When every candidate intersects the keyboard, no swipe is dispatched and abort is KEYBOARD_BLOCKED.
        """

        dispatcher = _RecordingDispatcher(outcomes=[])
        coordinator = SwipeRetryCoordinator(planner=SwipeRetryPlanner(), dispatcher=dispatcher)
        keyboard = KeyboardObservation(
            visibility=KeyboardVisibility.VISIBLE,
            bounds=Bounds(
                x=0,
                y=400,
                width=1080,
                height=1800,
                coordinate_system=CoordinateSystem.DEVICE_PIXEL,
            ),
        )

        execution = await coordinator.execute(
            original=self.__upward_path(),
            bounds=self.__viewport(),
            policy=SwipeRetryPolicy(enabled=True),
            keyboard=keyboard,
            original_before="A",
        )

        self.assertEqual(len(execution.attempts), 0)
        self.assertEqual(execution.aborted_for, AbortReason.KEYBOARD_BLOCKED)
        self.assertEqual(len(dispatcher.received), 0)

    async def test_iterates_retries_when_visual_unchanged(self) -> None:
        """
        Visual no-change forces the coordinator to dispatch the next retry candidate.
        """

        stuck = SwipeAttempt(
            index=0,
            path=self.__upward_path(),
            device=DeviceOutcome(succeeded=True),
            visual=VisualOutcome(changed=False, before="A", after="A"),
        )
        success = SwipeAttempt(
            index=1,
            path=self.__upward_path(),
            device=DeviceOutcome(succeeded=True),
            visual=VisualOutcome(changed=True, before="A", after="C"),
        )
        dispatcher = _RecordingDispatcher(outcomes=[stuck, success])
        coordinator = SwipeRetryCoordinator(planner=SwipeRetryPlanner(), dispatcher=dispatcher)

        execution = await coordinator.execute(
            original=self.__upward_path(),
            bounds=self.__viewport(),
            policy=SwipeRetryPolicy(
                enabled=True,
                direction=RetryDirection.INWARD,
                magnitudes=(0.10,),
            ),
            keyboard=KeyboardObservation(visibility=KeyboardVisibility.HIDDEN),
            original_before="A",
        )

        self.assertEqual(len(execution.attempts), 2)
        self.assertTrue(execution.effective)
        self.assertIsNone(execution.aborted_for)

    async def test_no_visual_change_after_all_attempts(self) -> None:
        """
        When no attempt produces visual change, abort reason is NO_VISUAL_CHANGE.
        """

        stuck = SwipeAttempt(
            index=0,
            path=self.__upward_path(),
            device=DeviceOutcome(succeeded=True),
            visual=VisualOutcome(changed=False, before="A", after="A"),
        )
        dispatcher = _RecordingDispatcher(outcomes=[stuck, stuck])
        coordinator = SwipeRetryCoordinator(planner=SwipeRetryPlanner(), dispatcher=dispatcher)

        execution = await coordinator.execute(
            original=self.__upward_path(),
            bounds=self.__viewport(),
            policy=SwipeRetryPolicy(
                enabled=True,
                direction=RetryDirection.INWARD,
                magnitudes=(0.10,),
            ),
            keyboard=KeyboardObservation(visibility=KeyboardVisibility.HIDDEN),
            original_before="A",
        )

        self.assertEqual(len(execution.attempts), 2)
        self.assertFalse(execution.effective)
        self.assertEqual(execution.aborted_for, AbortReason.NO_VISUAL_CHANGE)
