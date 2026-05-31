from __future__ import annotations

import unittest

from fathom.constants import ActionType
from fathom.core.agent.ladder import LoopActionLadder
from fathom.schemas.screens import ScreenState
from fathom.schemas.state import LoopDetector


class LoopActionLadderTest(unittest.TestCase):
    """
    Pins for the mechanical recovery-action escalation ladder.
    """

    def test_first_attempt_returns_back(self) -> None:
        """
        The first ladder attempt must return a BACK recovery action.
        """

        detector = LoopDetector(max_recovery=3)
        action = LoopActionLadder().next(detector=detector)

        self.assertIsNotNone(action)
        assert action is not None
        self.assertEqual(action.action_type, ActionType.BACK)

    def test_second_attempt_returns_scroll(self) -> None:
        """
        The second ladder attempt must escalate to SCROLL for non-scroll loops.
        """

        detector = LoopDetector(max_recovery=3)
        LoopActionLadder().next(detector=detector)
        action = LoopActionLadder().next(detector=detector)

        self.assertIsNotNone(action)
        assert action is not None
        self.assertEqual(action.action_type, ActionType.SCROLL)

    def test_second_attempt_skips_scroll_when_scroll_caused_loop(self) -> None:
        """
        A loop already caused by scroll-like actions must not recover by
        issuing another blind viewport scroll.
        """

        detector = LoopDetector(max_recovery=3)
        detector.record(
            screen=self.__screen(),
            action_type=ActionType.SWIPE_UP.value,
            action_description="Swipe up on feed",
        )

        ladder = LoopActionLadder()
        ladder.next(detector=detector)
        action = ladder.next(detector=detector)

        self.assertIsNotNone(action)
        assert action is not None
        self.assertEqual(action.action_type, ActionType.HOME)

    def test_detector_exposes_last_action_type(self) -> None:
        """
        The ladder depends on typed detector history instead of peeking
        into private deque state.
        """

        detector = LoopDetector(max_recovery=3)
        self.assertIsNone(detector.last_action_type)

        detector.record(
            screen=self.__screen(),
            action_type=ActionType.TAP.value,
            action_description="Tap continue",
        )

        self.assertEqual(detector.last_action_type, ActionType.TAP.value)

    def test_third_attempt_returns_home(self) -> None:
        """
        The third and later ladder attempts must escalate to HOME.
        """

        detector = LoopDetector(max_recovery=3)
        ladder = LoopActionLadder()
        ladder.next(detector=detector)
        ladder.next(detector=detector)
        action = ladder.next(detector=detector)

        self.assertIsNotNone(action)
        assert action is not None
        self.assertEqual(action.action_type, ActionType.HOME)

    def test_returns_none_when_detector_exhausted(self) -> None:
        """
        Once the detector cannot recover further, the ladder returns None.
        """

        detector = LoopDetector(max_recovery=1)
        ladder = LoopActionLadder()
        ladder.next(detector=detector)

        self.assertIsNone(ladder.next(detector=detector))

    @staticmethod
    def __screen() -> ScreenState:
        """
        Return a minimal stable screen for detector history.
        """

        return ScreenState(
            activity="app",
            timestamp=0,
            activity_hash="a" * 16,
            visual_hash="b" * 16,
        )
