from __future__ import annotations

import unittest

from fathom.constants import ActionType
from fathom.core.recovery.ladder import RecoveryActionLadder
from fathom.schemas.state import LoopDetector


class RecoveryActionLadderTest(unittest.TestCase):
    """
    Pins for the mechanical recovery-action escalation ladder.
    """

    def test_first_attempt_returns_back(self) -> None:
        """
        The first ladder attempt must return a BACK recovery action.
        """

        detector = LoopDetector(max_recovery=3)
        action = RecoveryActionLadder().next(detector=detector)

        self.assertIsNotNone(action)
        assert action is not None
        self.assertEqual(action.action_type, ActionType.BACK)

    def test_second_attempt_returns_scroll(self) -> None:
        """
        The second ladder attempt must escalate to a SCROLL action.
        """

        detector = LoopDetector(max_recovery=3)
        RecoveryActionLadder().next(detector=detector)
        action = RecoveryActionLadder().next(detector=detector)

        self.assertIsNotNone(action)
        assert action is not None
        self.assertEqual(action.action_type, ActionType.SCROLL)

    def test_third_attempt_returns_home(self) -> None:
        """
        The third and later ladder attempts must escalate to HOME.
        """

        detector = LoopDetector(max_recovery=3)
        ladder = RecoveryActionLadder()
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
        ladder = RecoveryActionLadder()
        ladder.next(detector=detector)

        self.assertIsNone(ladder.next(detector=detector))
