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
            timestamp=0,
            activity_hash="0" * 16,
            visual_hash=visual_hash,
            activity="com.example/.Main",
        )

    @staticmethod
    def __record_inert_streak(
        *,
        count: int,
        action: str,
        detector: LoopDetector,
    ) -> None:
        """
        Drive ``count`` records of ``(action, NO_PROGRESS)`` into the detector.
        """

        for _ in range(count):
            detector.record(
                action_type="tap",
                action_description=action,
                effect_status=ActionEffectStatus.NO_PROGRESS,
                screen=TestLoopDetectorInertRepetition.__screen(),
            )

    def test_fresh_detector_is_not_stuck(self) -> None:
        """
        A detector with no recorded history must report not stuck.
        """

        detector = LoopDetector()

        assert detector.is_stuck() is False

    def test_threshold_inert_same_action_marks_stuck(self) -> None:
        """
        Threshold-many identical action descriptors paired with trailing ``NO_PROGRESS`` effects must trip the detector
        independent of whether the classic screen / action repetition thresholds have accumulated enough samples.
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
            action_type="tap",
            screen=self.__screen(),
            action_description=action,
            effect_status=ActionEffectStatus.NO_PROGRESS,
        )
        detector.record(
            action_type="tap",
            screen=self.__screen(),
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
            action_type="tap",
            screen=self.__screen(),
            action_description="Tap A",
            effect_status=ActionEffectStatus.NO_PROGRESS,
        )
        detector.record(
            action_type="tap",
            screen=self.__screen(),
            action_description="Tap B",
            effect_status=ActionEffectStatus.NO_PROGRESS,
        )

        assert detector.is_stuck() is False

    def test_uncertain_effect_does_not_trip_detector(self) -> None:
        """
        ``UNCERTAIN`` is not ``NO_PROGRESS``; the detector must require the strong signal before declaring inert repetition.
        Each record uses a distinct screen hash so the unrelated screen-repetition detector cannot fire on the same history.
        """

        detector = LoopDetector()
        action = "Tap on Confirm & proceed button"
        hashes = ("0" * 16, "a" * 16, "f" * 16)

        for index in range(DEFAULT_INERT_REPETITION_THRESHOLD):
            detector.record(
                action_type="tap",
                action_description=action,
                effect_status=ActionEffectStatus.UNCERTAIN,
                screen=self.__screen(visual_hash=hashes[index]),
            )

        assert detector.is_stuck() is False

    def test_missing_effect_status_does_not_trip_detector(self) -> None:
        """
        Legacy ``record()`` calls without an effect status must not false-fire the inert detector
        empty slots are treated as "not classifiable", not "no progress". Distinct screen hashes keep the screen-repetition detector silent.
        """

        detector = LoopDetector()
        action = "Tap on Confirm & proceed button"
        hashes = ("0" * 16, "a" * 16, "f" * 16)

        for index in range(DEFAULT_INERT_REPETITION_THRESHOLD):
            detector.record(
                action_type="tap",
                action_description=action,
                screen=self.__screen(visual_hash=hashes[index]),
            )

        assert detector.is_stuck() is False

    def test_inert_streak_survives_checkpoint_round_trip(self) -> None:
        """
        After ``restore``, the detector's inert-repetition history must reproduce the same verdict as the live detector
        the new ``effect_statuses`` field on :class:`LoopDetectorState` is the field that keeps resumed runs aware of inert streaks.
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
