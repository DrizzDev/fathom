from __future__ import annotations

import unittest

from fathom.constants.runtime import DEFAULT_INERT_REPETITION_THRESHOLD
from fathom.schemas.effect import ActionEffectStatus
from fathom.schemas.loop import LoopEvidence, LoopReason
from fathom.schemas.screens import ScreenState
from fathom.schemas.state import LoopDetector
from fathom.schemas.vision import ActionKind


class LoopEvidenceTest(unittest.TestCase):
    """
    Pins the typed evidence snapshot returned by :class:`LoopDetector`.
    """

    @staticmethod
    def __screen(*, visual_hash: str = "a" * 16) -> ScreenState:
        """
        Returns ScreenState
        """
        return ScreenState(
            timestamp=0,
            xml_hash="x",
            activity_hash="ah",
            interaction_hash="i",
            visual_hash=visual_hash,
            activity="com.example/.Main",
        )

    def test_empty_detector_returns_not_stuck(self) -> None:
        """
        A detector with no records reports ``stuck=False`` and empty windows.
        """

        evidence = LoopDetector().evidence()

        self.assertFalse(evidence.stuck)

        self.assertEqual(evidence.recent, ())
        self.assertEqual(evidence.since_progress, ())
        self.assertIsInstance(evidence, LoopEvidence)
        self.assertIs(evidence.reason, LoopReason.NOT_STUCK)

    def test_recent_window_uses_action_kind_derivation(self) -> None:
        """
        Each ``LoopTurn`` carries the :class:`ActionKind` derived from action_type.
        """

        detector = LoopDetector()
        detector.record(
            screen=self.__screen(),
            action_type="validate",
            action_description="validate srp",
            effect_status=ActionEffectStatus.NO_PROGRESS,
        )
        detector.record(
            screen=self.__screen(),
            action_type="tap",
            action_description="tap button",
            effect_status=ActionEffectStatus.PROGRESS,
        )

        evidence = detector.evidence()

        self.assertEqual(len(evidence.recent), 2)
        self.assertIs(evidence.recent[0].action_kind, ActionKind.VALIDATION)
        self.assertIs(evidence.recent[1].action_kind, ActionKind.NAVIGATION)

    def test_since_progress_excludes_turns_before_last_progress(self) -> None:
        """
        Trailing slice starts AFTER the most recent PROGRESS effect.
        """

        detector = LoopDetector()
        detector.record(
            screen=self.__screen(visual_hash="a" * 16),
            action_type="tap",
            action_description="old tap",
            effect_status=ActionEffectStatus.NO_PROGRESS,
        )
        detector.record(
            screen=self.__screen(visual_hash="b" * 16),
            action_type="swipe_up",
            action_description="real progress",
            effect_status=ActionEffectStatus.PROGRESS,
        )
        detector.record(
            screen=self.__screen(visual_hash="c" * 16),
            action_type="validate",
            action_description="v1",
            effect_status=ActionEffectStatus.NO_PROGRESS,
        )
        detector.record(
            screen=self.__screen(visual_hash="c" * 16),
            action_type="validate",
            action_description="v2",
            effect_status=ActionEffectStatus.NO_PROGRESS,
        )

        evidence = detector.evidence()
        self.assertEqual(len(evidence.recent), 4)

        # since_progress excludes the old NO_PROGRESS tap and the PROGRESS swipe
        self.assertEqual(len(evidence.since_progress), 2)
        for turn in evidence.since_progress:
            self.assertIs(turn.action_kind, ActionKind.VALIDATION)
            self.assertIs(turn.effect_status, ActionEffectStatus.NO_PROGRESS)

    def test_since_progress_equals_recent_when_window_has_no_progress(self) -> None:
        """
        When no PROGRESS turn exists in the window the slice spans the full window.
        """

        detector = LoopDetector()
        for _ in range(3):
            detector.record(
                screen=self.__screen(),
                action_type="validate",
                action_description="v",
                effect_status=ActionEffectStatus.NO_PROGRESS,
            )

        evidence = detector.evidence()

        self.assertEqual(len(evidence.recent), 3)
        self.assertEqual(len(evidence.since_progress), 3)

    def test_unknown_effect_status_maps_to_uncertain(self) -> None:
        """
        Legacy turns without a recorded effect status surface as UNCERTAIN.
        """

        detector = LoopDetector()
        detector.record(
            action_type="tap",
            screen=self.__screen(),
            action_description="t",
        )

        evidence = detector.evidence()

        self.assertEqual(len(evidence.recent), 1)
        self.assertIs(evidence.recent[0].effect_status, ActionEffectStatus.UNCERTAIN)

    def test_reason_reports_inert_repetition_when_classifier_fires(self) -> None:
        """
        ``DEFAULT_INERT_REPETITION_THRESHOLD`` identical NO_PROGRESS records trip the inert-repetition detector.
        """

        detector = LoopDetector()

        for _ in range(DEFAULT_INERT_REPETITION_THRESHOLD):
            detector.record(
                screen=self.__screen(),
                action_type="validate",
                action_description="v",
                effect_status=ActionEffectStatus.NO_PROGRESS,
            )

        evidence = detector.evidence()

        self.assertTrue(evidence.stuck)
        self.assertIs(evidence.reason, LoopReason.INERT_REPETITION)

    def test_screen_hash_prefix_truncated_to_eight_characters(self) -> None:
        """
        Diagnostic hash prefix is kept short for log readability.
        """

        detector = LoopDetector()
        detector.record(
            action_type="tap",
            action_description="t",
            screen=self.__screen(visual_hash="abcdefghij" * 2),
        )

        evidence = detector.evidence()
        self.assertEqual(evidence.recent[0].screen_hash_prefix, "abcdefgh")

    def test_snapshot_tail_aligns_when_advance_clears_screens(self) -> None:
        """
        Regression: :meth:`advance` clears screens but preserves actions, so the deque's diverge.
        The snapshot must align from the tail or the gate will evaluate stale turns from before the most recent recovery.
        """

        detector = LoopDetector()

        # Drive an extensive action history so the actions deque is longer than the screens deque after the upcoming advance.
        for index in range(5):
            detector.record(
                action_type="tap",
                action_description=f"tap_{index}",
                effect_status=ActionEffectStatus.PROGRESS,
                screen=self.__screen(visual_hash=f"{index:016x}"),
            )

        # A real PROGRESS swipe triggers advance() in production via
        # observe_screen; simulate by calling advance directly so we land on
        # the same deque-misalignment as the production scenario.
        detector.advance()

        # Two consecutive validates with NO_PROGRESS on the same screen.
        validate_screen = self.__screen(visual_hash="9fc3e0c0deadbeef")
        for _ in range(2):
            detector.record(
                screen=validate_screen,
                action_type="validate",
                action_description="validate srp",
                effect_status=ActionEffectStatus.NO_PROGRESS,
            )

        evidence = detector.evidence()

        # Recent must reflect the 2 trailing validates, not the older taps.
        self.assertEqual(len(evidence.recent), 2)
        for turn in evidence.recent:
            self.assertIs(turn.action_kind, ActionKind.VALIDATION)
            self.assertIs(turn.effect_status, ActionEffectStatus.NO_PROGRESS)

        # since_progress is the full 2-turn slice (no PROGRESS in it).
        self.assertEqual(len(evidence.since_progress), 2)
        for turn in evidence.since_progress:
            self.assertIs(turn.action_kind, ActionKind.VALIDATION)
            self.assertIs(turn.effect_status, ActionEffectStatus.NO_PROGRESS)

    def test_evidence_does_not_mutate_detector(self) -> None:
        """
        Calling :meth:`evidence` twice in a row returns equivalent snapshots.
        """

        detector = LoopDetector()
        for _ in range(2):
            detector.record(
                screen=self.__screen(),
                action_type="validate",
                action_description="v",
                effect_status=ActionEffectStatus.NO_PROGRESS,
            )

        first = detector.evidence()
        second = detector.evidence()
        self.assertEqual(first.recent, second.recent)
        self.assertEqual(first.reason, second.reason)
        self.assertEqual(first.since_progress, second.since_progress)
