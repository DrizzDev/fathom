from __future__ import annotations

import unittest

from fathom.constants import ActionType
from fathom.core.runtime import RuntimeState
from fathom.core.supervision import RuntimeSupervisor
from fathom.schemas.actions import Action
from fathom.schemas.effect import ActionEffect, ActionEffectStatus
from fathom.schemas.localization import LocalizationResult, LocalizationStatus
from fathom.schemas.observation import KeyboardObservation, ScreenObservation
from fathom.schemas.screens import ScreenHashBundle
from fathom.schemas.supervision import BlockReason, VerdictKind


class RuntimeSupervisorTest(unittest.TestCase):
    """
    Covers pre-execution runtime supervision policies.
    """

    @staticmethod
    def __screen(*, keyboard_visible: bool = False) -> ScreenObservation:
        """
        Build a minimal screen observation.
        """

        return ScreenObservation(
            activity="app",
            hashes=ScreenHashBundle(
                visual_hash="0" * 16,
                xml_hash="0" * 16,
                interaction_hash="0" * 16,
            ),
            elements=(),
            keyboard=KeyboardObservation(visible=keyboard_visible),
        )

    @staticmethod
    def __resolved() -> LocalizationResult:
        """
        Build a resolved localization result.
        """

        return LocalizationResult(status=LocalizationStatus.RESOLVED, confidence=1.0)

    @staticmethod
    def __action(action_type: ActionType = ActionType.SWIPE_UP) -> Action:
        """
        Build an action for supervision.
        """

        return Action(
            action_type=action_type,
            target="page",
            rationale="test",
            confidence=1.0,
        )

    def test_keyboard_blocks_scroll(self) -> None:
        """
        Scroll/swipe actions are blocked while keyboard is visible.
        """

        verdict = RuntimeSupervisor.create().supervise(
            action=self.__action(),
            localization=self.__resolved(),
            observation=self.__screen(keyboard_visible=True),
            runtime=RuntimeState.create(),
        )

        self.assertEqual(verdict.kind, VerdictKind.BLOCK)
        self.assertEqual(verdict.reason, BlockReason.KEYBOARD_OCCLUDING)

    def test_repeated_no_effect_blocks_before_next_action(self) -> None:
        """
        Consecutive no-progress outcomes block the next repeated attempt.
        """

        runtime = RuntimeState.create()
        for _ in range(3):
            runtime.effects.record_effect(
                effect=ActionEffect(
                    status=ActionEffectStatus.NO_PROGRESS,
                    visual_progress=0.0,
                    phash_distance=0,
                )
            )

        verdict = RuntimeSupervisor.create().supervise(
            action=self.__action(action_type=ActionType.TAP),
            localization=self.__resolved(),
            observation=self.__screen(),
            runtime=runtime,
        )

        self.assertEqual(verdict.kind, VerdictKind.BLOCK)
        self.assertEqual(verdict.reason, BlockReason.REPEATED_NO_EFFECT)

    def test_safety_keywords_are_not_evaluated_on_runtime_path(self) -> None:
        """
        Destructive-keyword screening is intent-level, not per-action.

        The runtime supervisor must allow actions whose rationale or
        target text contains keywords from the safety vocabulary —
        intent-level screening happens once before workflow start via
        :class:`IntentSafetyClassifier`. Re-applying the substring scan
        per step produced false positives on every ``swipe`` (``"wipe"
        in "swipe"``) and blocked all scroll gestures.
        """

        runtime = RuntimeState.create()
        verdict = RuntimeSupervisor.create().supervise(
            action=Action(
                action_type=ActionType.TAP,
                target="factory reset",
                rationale="proceed",
                confidence=0.9,
            ),
            localization=self.__resolved(),
            observation=self.__screen(),
            runtime=runtime,
        )

        self.assertEqual(verdict.kind, VerdictKind.ALLOW)
        self.assertIsNone(verdict.reason)

    def test_target_unresolved_block_does_not_record_failure(self) -> None:
        """
        TARGET_UNRESOLVED blocks must not poison the action in failure memory.
        """

        runtime = RuntimeState.create()
        verdict = RuntimeSupervisor.create().supervise(
            action=self.__action(action_type=ActionType.TAP),
            localization=LocalizationResult(
                status=LocalizationStatus.UNRESOLVED,
                confidence=0.0,
            ),
            observation=self.__screen(),
            runtime=runtime,
        )

        self.assertEqual(verdict.kind, VerdictKind.BLOCK)
        self.assertEqual(verdict.reason, BlockReason.TARGET_UNRESOLVED)
        self.assertEqual(len(runtime.failures.records()), 0)

    def test_target_ambiguous_block_does_not_record_failure(self) -> None:
        """
        TARGET_AMBIGUOUS blocks must not poison the action in failure memory.
        """

        runtime = RuntimeState.create()
        verdict = RuntimeSupervisor.create().supervise(
            action=self.__action(action_type=ActionType.TAP),
            localization=LocalizationResult(
                status=LocalizationStatus.AMBIGUOUS,
                confidence=0.4,
            ),
            observation=self.__screen(),
            runtime=runtime,
        )

        self.assertEqual(verdict.kind, VerdictKind.BLOCK)
        self.assertEqual(verdict.reason, BlockReason.TARGET_AMBIGUOUS)
        self.assertEqual(len(runtime.failures.records()), 0)

    def test_keyboard_block_records_failure(self) -> None:
        """
        Recorded reasons (KEYBOARD_OCCLUDING) must be captured in failure memory.
        """

        runtime = RuntimeState.create()
        RuntimeSupervisor.create().supervise(
            action=self.__action(),
            localization=self.__resolved(),
            observation=self.__screen(keyboard_visible=True),
            runtime=runtime,
        )

        self.assertEqual(len(runtime.failures.records()), 1)
        self.assertEqual(runtime.failures.records()[0].reason, BlockReason.KEYBOARD_OCCLUDING)
