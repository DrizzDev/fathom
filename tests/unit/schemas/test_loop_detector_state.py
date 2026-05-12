"""
Unit tests for :class:`LoopDetectorState` round-tripping. Pins the
serialization seam that survives checkpoint restore so loop-detection
evidence is not silently wiped between graph iterations.
"""

from __future__ import annotations

from fathom.constants.screen import LOOP_DETECTOR_WINDOW_SIZE
from fathom.schemas.screens import ScreenState
from fathom.schemas.state import LoopDetector, LoopDetectorState


class TestLoopDetectorState:
    """
    Behavioural pins for to-state / restore round-trips.
    """

    @staticmethod
    def __screen(*, visual_hash: str = "a" * 16) -> ScreenState:
        """
        Minimal valid :class:`ScreenState` for snapshot tests.
        """

        return ScreenState(
            activity="com.example/.Main",
            timestamp=0,
            activity_hash="0" * 16,
            visual_hash=visual_hash,
        )

    def test_empty_detector_round_trips_cleanly(self) -> None:
        """
        A fresh detector snapshot deserializes back to an empty detector.
        """

        original = LoopDetector()
        state = original.to_state()
        restored = LoopDetector()
        restored.restore(state=state)

        assert restored.to_state() == state

    def test_record_then_round_trip_preserves_entries(self) -> None:
        """
        Recording into the detector and round-tripping through the
        snapshot must preserve every deque entry and the recovery count.
        """

        original = LoopDetector()
        original.record(screen=self.__screen(visual_hash="1" * 16), action_description="tap A")
        original.record(screen=self.__screen(visual_hash="2" * 16), action_description="tap B")
        original.record_recovery_attempt()

        snapshot = original.to_state()

        restored = LoopDetector()
        restored.restore(state=snapshot)
        re_snapshot = restored.to_state()

        assert re_snapshot.actions == snapshot.actions
        assert re_snapshot.hashes == snapshot.hashes
        assert re_snapshot.recovery_attempts == snapshot.recovery_attempts
        assert len(re_snapshot.screens) == len(snapshot.screens)

    def test_window_size_controls_deque_maxlen_on_fresh_detector(self) -> None:
        """
        A fresh detector must size its internal deques from
        ``window_size`` — not a hardcoded default.
        """

        detector = LoopDetector(window_size=5)
        for index in range(10):
            detector.record(
                screen=self.__screen(visual_hash=f"{index:016d}"), action_description=f"x{index}"
            )

        snapshot = detector.to_state()
        assert len(snapshot.actions) == 5
        assert len(snapshot.hashes) == 5

    def test_default_window_size_matches_constant(self) -> None:
        """
        Default ``window_size`` must come from the named constant.
        """

        assert LoopDetector().window_size == LOOP_DETECTOR_WINDOW_SIZE

    def test_state_drops_extra_entries_to_window_size(self) -> None:
        """
        Restoring a snapshot longer than ``window_size`` must truncate to
        the configured maxlen rather than blow past it.
        """

        oversized = LoopDetectorState(
            actions=[f"act-{index}" for index in range(20)],
            types=[f"type-{index}" for index in range(20)],
            hashes=[f"h-{index}" for index in range(20)],
            screens=[],
            timestamps=[float(index) for index in range(20)],
            recovery_attempts=2,
        )

        detector = LoopDetector(window_size=5)
        detector.restore(state=oversized)
        snapshot = detector.to_state()

        assert len(snapshot.actions) == 5
        assert snapshot.actions[-1] == "act-19"
        assert snapshot.recovery_attempts == 2
