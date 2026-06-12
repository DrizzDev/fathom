"""
Unit tests for the loop detector.
"""

from __future__ import annotations

from fathom.schemas.state import LoopDetector, ScreenState


class TestLoopDetector:
    """
    Stuck-state detection and content-exhaustion handling.
    """

    @staticmethod
    def __screen(
        *,
        activity: str = "TestActivity",
        visual: str = "a1b2c3d4",
        activity_hash: str = "deadbeef",
    ) -> ScreenState:
        return ScreenState(
            activity=activity,
            timestamp=1000,
            activity_hash=activity_hash,
            structural_hash="c0ffee",
            visual_hash=visual,
        )

    def test_exhaustion_signal_clears_stuck(self) -> None:
        detector = LoopDetector(threshold=3)
        screen = self.__screen()
        for _ in range(3):
            detector.record(screen, "swipe_left")
        assert detector.is_stuck() is True

        detector.signal_content_exhausted()
        assert detector.is_stuck() is False

    def test_exhaustion_clears_history(self) -> None:
        detector = LoopDetector(threshold=3)
        screen = self.__screen()
        detector.record(screen, "swipe_left")
        detector.signal_content_exhausted()

        # After the clear, two more identical records are below the threshold.
        detector.record(screen, "swipe_left")
        detector.record(screen, "swipe_left")
        assert detector.is_stuck() is False

    def test_repeated_action_triggers_stuck(self) -> None:
        detector = LoopDetector(threshold=3)
        for i in range(3):
            detector.record(
                self.__screen(
                    activity=f"TestActivity{i}", visual=f"a1b2c3d{i}", activity_hash=f"deadbe0{i}"
                ),
                "tap:retry-button",
            )
        assert detector.is_stuck() is True

    def test_diverse_actions_avoid_false_positive(self) -> None:
        detector = LoopDetector(threshold=3)
        screen = self.__screen()
        detector.record(screen, "tap:menu")
        detector.record(screen, "scroll:down")
        detector.record(screen, "back")
        assert detector.is_stuck() is False
