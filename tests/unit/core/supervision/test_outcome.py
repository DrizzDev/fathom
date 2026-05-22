from __future__ import annotations

import unittest
from typing import Tuple

from fathom.constants import ActionType
from fathom.constants.scroll import ScrollEvidenceSource, ScrollVerdictKind
from fathom.core.supervision import OutcomeClassifier
from fathom.schemas.actions import Action, Bounds, CoordinateSystem
from fathom.schemas.observation import (
    KeyboardObservation,
    OverlayObservation,
    ScreenObservation,
)
from fathom.schemas.outcomes import OutcomeStatus
from fathom.schemas.screens import ScreenDiff, ScreenHashBundle
from fathom.schemas.scroll import ScrollOutcome, ScrollVerdict


class OutcomeClassifierTest(unittest.TestCase):
    """
    Pins :class:`OutcomeClassifier` rules.

    The classifier owns the per-action-type decision tree that turns a
    screen diff plus before/after observations into a typed
    :class:`ActionOutcome`. Tests cover the device-failure short-circuit,
    the unavailable-evidence path, the tap overlay-state heuristic,
    the type/scroll/generic diff rules, and the keyboard-blocks-scroll
    safety rule.
    """

    @staticmethod
    def __hashes() -> ScreenHashBundle:
        """
        Deterministic :class:`ScreenHashBundle`. The classifier does not
        compare hashes; the bundle exists only to satisfy the schema.
        """

        return ScreenHashBundle(
            visual_hash="0" * 16,
            xml_hash="a" * 16,
            interaction_hash="b" * 16,
        )

    @classmethod
    def __observation(
        cls,
        *,
        overlays: Tuple[OverlayObservation, ...] = (),
        keyboard_visible: bool = False,
    ) -> ScreenObservation:
        """
        :class:`ScreenObservation` fixture parameterised on overlay
        presence (drives the tap-overlay-state branch) and keyboard
        visibility (drives the scroll-blocked-by-keyboard branch).
        """

        return ScreenObservation(
            activity="app",
            elements=(),
            hashes=cls.__hashes(),
            overlays=overlays,
            keyboard=KeyboardObservation(visible=keyboard_visible),
        )

    @staticmethod
    def __overlay_observation() -> OverlayObservation:
        """
        Single :class:`OverlayObservation` fixture used to flip the
        before/after overlay count in the tap-overlay-state test.
        """

        return OverlayObservation(
            visible=True,
            bounds=Bounds(
                x=0,
                y=0,
                width=100,
                height=100,
                coordinate_system=CoordinateSystem.DEVICE_PIXEL,
            ),
            candidates=(),
        )

    @staticmethod
    def __action(*, action_type: ActionType = ActionType.TAP) -> Action:
        """
        :class:`Action` fixture parameterised on type so each test can
        drive the relevant classifier branch (tap / type / scroll / generic).
        """

        return Action(
            action_type=action_type,
            target="Continue",
            rationale="t",
            confidence=1.0,
        )

    @staticmethod
    def __diff(
        *,
        action_had_effect: bool,
        is_genuinely_different_state: bool = False,
    ) -> ScreenDiff:
        """
        :class:`ScreenDiff` fixture parameterised on the two properties
        the classifier inspects. Other fields are driven coherently so
        the diff looks realistic regardless of branch under test.
        """

        if action_had_effect:
            return ScreenDiff(
                phash_distance=20,
                ssim_score=0.5,
                content_pixel_diff_ratio=0.5 if is_genuinely_different_state else 0.05,
                xml_hash_changed=True,
                interaction_hash_changed=True,
                activity_changed=is_genuinely_different_state,
            )

        return ScreenDiff(
            phash_distance=0,
            ssim_score=1.0,
            content_pixel_diff_ratio=0.0,
            xml_hash_changed=False,
            interaction_hash_changed=False,
            activity_changed=False,
        )

    def test_failed_execution_without_post_action_evidence_returns_blocked_outcome(self) -> None:
        """
        ``success=False`` with no post-action evidence remains BLOCKED.
        """

        outcome = OutcomeClassifier().classify(
            success=False,
            action=self.__action(),
            before=self.__observation(),
            diff=None,
            after=None,
        )

        self.assertEqual(outcome.status, OutcomeStatus.BLOCKED)

    def test_failed_scroll_with_strong_visual_progress_is_effective(self) -> None:
        """
        Failed execution must still classify EFFECTIVE when the diff proves a real scroll landed.
        """

        outcome = OutcomeClassifier().classify(
            success=False,
            action=self.__action(action_type=ActionType.SWIPE_UP),
            before=self.__observation(),
            diff=self.__diff(
                action_had_effect=True,
                is_genuinely_different_state=True,
            ),
            after=self.__observation(),
            scroll_outcome=ScrollOutcome(
                success=False,
                final=ScrollVerdict(
                    kind=ScrollVerdictKind.AMBIGUOUS,
                    source=ScrollEvidenceSource.CORRELATION,
                    confidence=0.55,
                    distance=0,
                    detail="translation_in_uncertain_band",
                ),
            ),
        )

        self.assertEqual(outcome.status, OutcomeStatus.EFFECTIVE)

    def test_missing_after_observation_returns_unknown(self) -> None:
        """
        Successful command with no post-action observation yields
        UNKNOWN — the classifier cannot prove effect or no-effect.
        """

        outcome = OutcomeClassifier().classify(
            success=True,
            action=self.__action(),
            before=self.__observation(),
            diff=None,
            after=None,
        )

        self.assertEqual(outcome.status, OutcomeStatus.UNKNOWN)

    def test_tap_with_overlay_state_change_is_effective(self) -> None:
        """
        A tap that flips the overlay count between before and after is
        effective even when the diff would otherwise report no effect —
        dismissing a dialog often produces a tiny pixel delta.
        """

        before = self.__observation(overlays=(self.__overlay_observation(),))
        after = self.__observation(overlays=())

        outcome = OutcomeClassifier().classify(
            success=True,
            action=self.__action(action_type=ActionType.TAP),
            before=before,
            diff=self.__diff(action_had_effect=False),
            after=after,
        )

        self.assertEqual(outcome.status, OutcomeStatus.EFFECTIVE)

    def test_tap_without_overlay_change_falls_back_to_diff(self) -> None:
        """
        A tap with no overlay change classifies from the diff signal —
        ``action_had_effect=True`` produces EFFECTIVE.
        """

        outcome = OutcomeClassifier().classify(
            success=True,
            action=self.__action(action_type=ActionType.TAP),
            before=self.__observation(),
            diff=self.__diff(action_had_effect=True),
            after=self.__observation(),
        )

        self.assertEqual(outcome.status, OutcomeStatus.EFFECTIVE)

    def test_type_with_effect_is_effective(self) -> None:
        """
        TYPE actions classify EFFECTIVE when the diff reports any UI
        change — typing into an input field always produces a visible
        delta.
        """

        outcome = OutcomeClassifier().classify(
            success=True,
            action=self.__action(action_type=ActionType.TYPE),
            before=self.__observation(),
            diff=self.__diff(action_had_effect=True),
            after=self.__observation(),
        )

        self.assertEqual(outcome.status, OutcomeStatus.EFFECTIVE)

    def test_type_without_effect_is_no_effect(self) -> None:
        """
        TYPE actions that produce no visible diff classify NO_EFFECT —
        the device accepted the keystrokes but the field did not focus.
        """

        outcome = OutcomeClassifier().classify(
            success=True,
            action=self.__action(action_type=ActionType.TYPE),
            before=self.__observation(),
            diff=self.__diff(action_had_effect=False),
            after=self.__observation(),
        )

        self.assertEqual(outcome.status, OutcomeStatus.NO_EFFECT)

    def test_scroll_with_keyboard_visible_is_no_effect(self) -> None:
        """
        Scroll attempted while the keyboard is up is NO_EFFECT even
        when the diff reports change — the keyboard occlusion makes any
        viewport delta unreliable evidence of scroll progress.
        """

        outcome = OutcomeClassifier().classify(
            success=True,
            action=self.__action(action_type=ActionType.SWIPE_UP),
            before=self.__observation(keyboard_visible=True),
            diff=self.__diff(
                action_had_effect=True,
                is_genuinely_different_state=True,
            ),
            after=self.__observation(),
        )

        self.assertEqual(outcome.status, OutcomeStatus.NO_EFFECT)

    def test_scroll_supervisor_progressed_is_advisory_when_diff_shows_no_effect(self) -> None:
        """
        Scroll diagnostics are advisory; the unified diff pipeline owns the final status.
        """

        outcome = OutcomeClassifier().classify(
            success=True,
            action=self.__action(action_type=ActionType.SWIPE_UP),
            before=self.__observation(),
            diff=self.__diff(action_had_effect=False),
            after=self.__observation(),
            scroll_outcome=ScrollOutcome(
                success=True,
                final=ScrollVerdict(
                    kind=ScrollVerdictKind.PROGRESSED,
                    source=ScrollEvidenceSource.CORRELATION,
                    confidence=0.99,
                    distance=320,
                ),
            ),
        )

        self.assertEqual(outcome.status, OutcomeStatus.NO_EFFECT)

    def test_scroll_supervisor_wrong_axis_does_not_override_visible_progress(self) -> None:
        """
        Strong visual progress remains effective even when the scroll diagnostic disagrees.
        """

        outcome = OutcomeClassifier().classify(
            success=True,
            action=self.__action(action_type=ActionType.SWIPE_UP),
            before=self.__observation(),
            diff=self.__diff(
                action_had_effect=True,
                is_genuinely_different_state=True,
            ),
            after=self.__observation(),
            scroll_outcome=ScrollOutcome(
                success=False,
                final=ScrollVerdict(
                    kind=ScrollVerdictKind.WRONG_AXIS,
                    source=ScrollEvidenceSource.CORRELATION,
                    confidence=0.98,
                    distance=280,
                ),
            ),
        )

        self.assertEqual(outcome.status, OutcomeStatus.EFFECTIVE)

    def test_scroll_supervisor_ambiguous_keeps_diff_owned_effective_status(self) -> None:
        """
        Ambiguous scroll diagnostics must not downgrade a clearly effective visual diff.
        """

        outcome = OutcomeClassifier().classify(
            success=True,
            action=self.__action(action_type=ActionType.SWIPE_UP),
            before=self.__observation(),
            diff=self.__diff(action_had_effect=True),
            after=self.__observation(),
            scroll_outcome=ScrollOutcome(
                success=False,
                final=ScrollVerdict(
                    kind=ScrollVerdictKind.AMBIGUOUS,
                    source=ScrollEvidenceSource.CORRELATION,
                    confidence=0.55,
                    distance=0,
                ),
            ),
        )

        self.assertEqual(outcome.status, OutcomeStatus.EFFECTIVE)

    def test_scroll_with_genuinely_different_state_is_effective(self) -> None:
        """
        Scroll classifies EFFECTIVE only on the stronger
        ``is_genuinely_different_state`` signal — the looser
        ``action_had_effect`` is not enough because pHash noise alone
        would otherwise pass the gate.
        """

        outcome = OutcomeClassifier().classify(
            success=True,
            action=self.__action(action_type=ActionType.SWIPE_UP),
            before=self.__observation(),
            diff=self.__diff(
                action_had_effect=True,
                is_genuinely_different_state=True,
            ),
            after=self.__observation(),
        )

        self.assertEqual(outcome.status, OutcomeStatus.EFFECTIVE)

    def test_scroll_without_genuine_change_is_no_effect(self) -> None:
        """
        Scroll on a screen that did not meaningfully change classifies
        NO_EFFECT — the loop detector relies on this verdict to spot
        bottom-of-list and dead-scroll situations.
        """

        outcome = OutcomeClassifier().classify(
            success=True,
            action=self.__action(action_type=ActionType.SWIPE_UP),
            before=self.__observation(),
            diff=self.__diff(action_had_effect=False),
            after=self.__observation(),
        )

        self.assertEqual(outcome.status, OutcomeStatus.NO_EFFECT)

    def test_generic_action_uses_diff_evidence(self) -> None:
        """
        BACK / HOME / WAIT (and any unknown action type) classify from
        ``action_had_effect`` directly — they have no type-specific
        heuristic and trust the diff signal.
        """

        outcome = OutcomeClassifier().classify(
            success=True,
            action=self.__action(action_type=ActionType.BACK),
            before=self.__observation(),
            diff=self.__diff(action_had_effect=True),
            after=self.__observation(),
        )

        self.assertEqual(outcome.status, OutcomeStatus.EFFECTIVE)

    def test_control_action_returns_unknown_without_participating_in_ui_effect_rules(self) -> None:
        """
        ASK_USER is a control-flow action, not a device-effect classification target.
        """

        outcome = OutcomeClassifier().classify(
            success=True,
            action=self.__action(action_type=ActionType.ASK_USER),
            before=self.__observation(),
            diff=self.__diff(action_had_effect=True),
            after=self.__observation(),
        )

        self.assertEqual(outcome.status, OutcomeStatus.UNKNOWN)
