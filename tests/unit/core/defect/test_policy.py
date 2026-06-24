from __future__ import annotations

import unittest
from typing import Any

from fathom.constants.defect import DefectSignal, DefectSource, DefectVerification
from fathom.constants.screen import HitOutcome, ScreenKind
from fathom.core.defect.policy import DeadTapVerificationPolicy, WebViewBlankPolicy
from fathom.schemas.actions import CoordinateSource
from fathom.schemas.defect import Defect, DefectEvidence, StepSignals


class DeadTapVerificationPolicyTest(unittest.TestCase):
    """
    Verifies the corroboration matrix that grades a dead tap confirmed or needs-review.
    """

    def setUp(self) -> None:
        self.__policy = DeadTapVerificationPolicy()

    @staticmethod
    def __signals(**overrides: Any) -> StepSignals:
        """
        Builds a fully-corroborated dead-tap signal, overriding one guard at a time.
        """

        base = {
            "screen": "hash-1",
            "activity": "com.app/.Home",
            "action_target": "Settings row",
            "expects_transition": True,
            "screen_changed": False,
            "post_activity": "com.app/.Home",
            "grounding": CoordinateSource.VISION,
            "confidence": 1.0,
            "target_hit": HitOutcome.HIT,
        }
        base.update(overrides)
        return StepSignals(**base)

    def test_well_corroborated_dead_tap_is_confirmed(self) -> None:
        verdict = self.__policy.verify(signals=self.__signals())
        self.assertEqual(verdict, DefectVerification.CONFIRMED)

    def test_blind_model_grounding_needs_review(self) -> None:
        verdict = self.__policy.verify(signals=self.__signals(grounding=CoordinateSource.MODEL))
        self.assertEqual(verdict, DefectVerification.NEEDS_REVIEW)

    def test_absent_grounding_needs_review(self) -> None:
        verdict = self.__policy.verify(signals=self.__signals(grounding=None))
        self.assertEqual(verdict, DefectVerification.NEEDS_REVIEW)

    def test_low_confidence_needs_review(self) -> None:
        verdict = self.__policy.verify(signals=self.__signals(confidence=0.4))
        self.assertEqual(verdict, DefectVerification.NEEDS_REVIEW)

    def test_activity_change_needs_review(self) -> None:
        verdict = self.__policy.verify(signals=self.__signals(post_activity="com.app/.Other"))
        self.assertEqual(verdict, DefectVerification.NEEDS_REVIEW)

    def test_missed_target_needs_review(self) -> None:
        verdict = self.__policy.verify(signals=self.__signals(target_hit=HitOutcome.MISS))
        self.assertEqual(verdict, DefectVerification.NEEDS_REVIEW)

    def test_unknown_hit_alone_does_not_downgrade(self) -> None:
        verdict = self.__policy.verify(signals=self.__signals(target_hit=HitOutcome.UNKNOWN))
        self.assertEqual(verdict, DefectVerification.CONFIRMED)

    def test_missing_post_activity_does_not_downgrade(self) -> None:
        verdict = self.__policy.verify(signals=self.__signals(post_activity=None))
        self.assertEqual(verdict, DefectVerification.CONFIRMED)


class WebViewBlankPolicyTest(unittest.TestCase):
    """
    Verifies that a blank WebView is held for review while other findings pass through.
    """

    @staticmethod
    def __defect(signal: DefectSignal) -> Defect:
        return Defect.from_signal(
            signal=signal,
            source=DefectSource.POST_RUN,
            summary="finding",
            evidence=DefectEvidence(screen="hash-1"),
        )

    def test_webview_empty_state_is_held_for_review(self) -> None:
        reviewed = WebViewBlankPolicy.review(
            defect=self.__defect(DefectSignal.EMPTY_STATE), kind=ScreenKind.WEBVIEW
        )
        self.assertEqual(reviewed.verification, DefectVerification.NEEDS_REVIEW)

    def test_webview_other_signal_is_unchanged(self) -> None:
        reviewed = WebViewBlankPolicy.review(
            defect=self.__defect(DefectSignal.CONTRAST), kind=ScreenKind.WEBVIEW
        )
        self.assertEqual(reviewed.verification, DefectVerification.CONFIRMED)

    def test_native_empty_state_is_unchanged(self) -> None:
        reviewed = WebViewBlankPolicy.review(
            defect=self.__defect(DefectSignal.EMPTY_STATE), kind=ScreenKind.NATIVE
        )
        self.assertEqual(reviewed.verification, DefectVerification.CONFIRMED)


if __name__ == "__main__":
    unittest.main()
