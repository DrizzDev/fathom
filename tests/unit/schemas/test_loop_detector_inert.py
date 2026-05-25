"""
Unit pins for :class:`LoopDetector` inert-action-repetition detection.

Detects the HmrHD-class stuck pattern: the agent re-proposes the same
action across consecutive turns while the effect classifier reports
NO_PROGRESS. Independent of the screen-history thresholds so the
planner can pivot one wasted action earlier than the classic detectors.
"""

from __future__ import annotations

from fathom.constants.runtime import DEFAULT_INERT_REPETITION_THRESHOLD
from fathom.schemas.effect import ActionEffectStatus
from fathom.schemas.screens import ScreenState
from fathom.schemas.state import LoopDetector


class TestLoopDetectorInertRepetition:
    """
    Pins the inert-action-repetition detector on :class:`LoopDetector`.
    """

    @staticmethod
    def __screen(*, visual_hash: str = "a" * 16) -> ScreenState:
        """
        Minimal valid :class:`ScreenState` fixture.
        """

        return ScreenState(
            activity="com.example/.Main",
            timestamp=0,
            activity_hash="0" * 16,
            visual_hash=visual_hash,
        )

    @staticmethod
    def __record_inert_streak(
        *,
        detector: LoopDetector,
        action: str,
        count: int,
    ) -> None:
        """
        Drive ``count`` records of ``(action, NO_PROGRESS)`` into the detector.
        """

        for _ in range(count):
            detector.record(
                screen=TestLoopDetectorInertRepetition.__screen(),
                action_type="tap",
                action_description=action,
                effect_status=ActionEffectStatus.NO_PROGRESS,
            )

    def test_fresh_detector_is_not_stuck(self) -> None:
        """
        A detector with no recorded history must report not stuck.
        """

        detector = LoopDetector()

        assert detector.is_stuck() is False

    def test_threshold_inert_same_action_marks_stuck(self) -> None:
        """
        Threshold-many identical action descriptors paired with trailing
        ``NO_PROGRESS`` effects must trip the detector — independent of
        whether the classic screen / action repetition thresholds have
        accumulated enough samples.
        """

        detector = LoopDetector()

        self.__record_inert_streak(
            detector=detector,
            action="Tap on Confirm & proceed button",
            count=DEFAULT_INERT_REPETITION_THRESHOLD,
        )

        assert detector.is_stuck() is True

    def test_progress_status_breaks_inert_streak(self) -> None:
        """
        A trailing ``PROGRESS`` effect must reset the inert streak so
        the detector no longer reports stuck on the same action.
        """

        detector = LoopDetector()
        action = "Tap on Confirm & proceed button"

        detector.record(
            screen=self.__screen(),
            action_type="tap",
            action_description=action,
            effect_status=ActionEffectStatus.NO_PROGRESS,
        )
        detector.record(
            screen=self.__screen(),
            action_type="tap",
            action_description=action,
            effect_status=ActionEffectStatus.PROGRESS,
        )

        assert detector.is_stuck() is False

    def test_different_actions_do_not_count_as_inert_repetition(self) -> None:
        """
        Distinct trailing action descriptors must not trip the detector
        even when both effects are ``NO_PROGRESS``.
        """

        detector = LoopDetector()

        detector.record(
            screen=self.__screen(),
            action_type="tap",
            action_description="Tap A",
            effect_status=ActionEffectStatus.NO_PROGRESS,
        )
        detector.record(
            screen=self.__screen(),
            action_type="tap",
            action_description="Tap B",
            effect_status=ActionEffectStatus.NO_PROGRESS,
        )

        assert detector.is_stuck() is False

    def test_uncertain_effect_does_not_trip_detector(self) -> None:
        """
        ``UNCERTAIN`` is not ``NO_PROGRESS``; the detector must require
        the strong signal before declaring inert repetition.
        """

        detector = LoopDetector()
        action = "Tap on Confirm & proceed button"

        for _ in range(DEFAULT_INERT_REPETITION_THRESHOLD):
            detector.record(
                screen=self.__screen(),
                action_type="tap",
                action_description=action,
                effect_status=ActionEffectStatus.UNCERTAIN,
            )

        assert detector.is_stuck() is False

    def test_missing_effect_status_does_not_trip_detector(self) -> None:
        """
        Legacy ``record()`` calls without an effect status must not
        false-fire the inert detector — empty slots are treated as
        "not classifiable", not "no progress".
        """

        detector = LoopDetector()
        action = "Tap on Confirm & proceed button"

        for _ in range(DEFAULT_INERT_REPETITION_THRESHOLD):
            detector.record(
                screen=self.__screen(),
                action_type="tap",
                action_description=action,
            )

        assert detector.is_stuck() is False

    def test_inert_streak_survives_checkpoint_round_trip(self) -> None:
        """
        After ``restore``, the detector's inert-repetition history must
        reproduce the same verdict as the live detector — the new
        ``effect_statuses`` field on :class:`LoopDetectorState` is the
        field that keeps resumed runs aware of inert streaks.
        """

        live = LoopDetector()
        self.__record_inert_streak(
            detector=live,
            action="Tap on Confirm & proceed button",
            count=DEFAULT_INERT_REPETITION_THRESHOLD,
        )
        assert live.is_stuck() is True

        restored = LoopDetector(window_size=live.window_size, threshold=live.threshold)
        restored.restore(state=live.to_state())

        assert restored.is_stuck() is True
