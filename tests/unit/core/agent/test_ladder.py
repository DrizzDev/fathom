from __future__ import annotations

import unittest

from fathom.constants import ActionType
from fathom.core.agent.ladder import LoopActionLadder
from fathom.core.agent.loop import LoopDetector
from fathom.schemas.capabilities import DeviceCapability
from fathom.schemas.screens import ScreenState


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

        ladder = LoopActionLadder()
        detector = LoopDetector(max_recovery=3)

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

        ladder = LoopActionLadder()
        detector = LoopDetector(max_recovery=1)

        ladder.next(detector=detector)

        self.assertIsNone(ladder.next(detector=detector))

    def test_ios_capability_skips_back_rung_first_attempt_returns_scroll(self) -> None:
        """
        qMrGC replay: an iOS device adapter that cannot dispatch BACK must
        cause the ladder to start at SCROLL instead, never emitting BACK.
        """

        detector = LoopDetector(max_recovery=3)
        ladder = LoopActionLadder(
            device=DeviceCapability(system_back_supported=False),
        )

        action = ladder.next(detector=detector)

        self.assertIsNotNone(action)
        assert action is not None
        self.assertEqual(action.action_type, ActionType.SCROLL)

    def test_ios_capability_second_attempt_returns_home(self) -> None:
        """
        With BACK filtered out, the ladder collapses to SCROLL then HOME.
        """

        detector = LoopDetector(max_recovery=3)
        ladder = LoopActionLadder(
            device=DeviceCapability(system_back_supported=False),
        )

        ladder.next(detector=detector)
        action = ladder.next(detector=detector)

        self.assertIsNotNone(action)
        assert action is not None
        self.assertEqual(action.action_type, ActionType.HOME)

    def test_ios_capability_with_scroll_loop_skips_to_home(self) -> None:
        """
        iOS device with an already-scrolling stuck signal: SCROLL is filtered
        at the rung level only by capability, but the runtime scroll-loop
        check still bypasses SCROLL — the agent goes straight to HOME.
        """

        detector = LoopDetector(max_recovery=3)
        detector.record(
            screen=self.__screen(),
            action_type="swipe_up",
            action_description="Swipe up on feed",
        )
        ladder = LoopActionLadder(
            device=DeviceCapability(system_back_supported=False),
        )

        action = ladder.next(detector=detector)

        self.assertIsNotNone(action)
        assert action is not None
        self.assertEqual(action.action_type, ActionType.HOME)

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


class LoopActionLadderValidatePassiveTest(unittest.TestCase):
    """
    Pin the passive-VALIDATE behavior: no mechanical recovery dispatched, no escalation to BACK or HOME.
    """

    def test_validate_last_action_yields_no_mechanical_recovery(self) -> None:
        """
        After a VALIDATE turn, the ladder emits no rung; agent re-plans next turn instead of being navigated away.
        """

        detector = LoopDetector(max_recovery=3)
        detector.record(
            screen=self.__screen(),
            action_type=ActionType.VALIDATE.value,
            action_description="Validate current state",
        )

        self.assertIsNone(LoopActionLadder().next(detector=detector))

    def test_validate_passive_still_increments_recovery_budget(self) -> None:
        """
        Recovery attempts still count toward max_recovery so unbounded passive-VALIDATE loops eventually exhaust.
        """

        detector = LoopDetector(max_recovery=2)
        detector.record(
            screen=self.__screen(),
            action_type=ActionType.VALIDATE.value,
            action_description="Validate current state",
        )
        ladder = LoopActionLadder()

        ladder.next(detector=detector)
        ladder.next(detector=detector)

        self.assertFalse(detector.can_recover())

    def test_validate_passive_on_no_back_device_does_not_emit_home(self) -> None:
        """
        A no-BACK device must not jump to HOME after a passive VALIDATE.
        """

        detector = LoopDetector(max_recovery=3)
        detector.record(
            screen=self.__screen(),
            action_type=ActionType.VALIDATE.value,
            action_description="Validate current state",
        )
        ladder = LoopActionLadder(device=DeviceCapability(system_back_supported=False))

        self.assertIsNone(ladder.next(detector=detector))

    def test_tap_last_action_keeps_normal_back_recovery(self) -> None:
        """
        A non-VALIDATE last action continues to receive the normal mechanical ladder (BACK first).
        """

        detector = LoopDetector(max_recovery=3)
        detector.record(
            screen=self.__screen(),
            action_type=ActionType.TAP.value,
            action_description="Tap submit",
        )

        action = LoopActionLadder().next(detector=detector)

        self.assertIsNotNone(action)
        assert action is not None
        self.assertEqual(action.action_type, ActionType.BACK)

    @staticmethod
    def __screen() -> ScreenState:
        """
        Return a minimal stable screen for detector history.
        """

        return ScreenState(
            timestamp=0,
            activity="app",
            visual_hash="b" * 16,
            activity_hash="a" * 16,
        )
