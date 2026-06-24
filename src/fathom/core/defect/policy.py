"""
Verification policy that grades inline dead-tap signals before they reach the report.
"""

from __future__ import annotations

from fathom.constants.defect import DEAD_TAP_MIN_CONFIDENCE, DefectVerification
from fathom.constants.screen import HitOutcome
from fathom.schemas.defect import StepSignals


class DeadTapVerificationPolicy:
    """
    Grades a dead-tap signal as confirmed only when its benign explanations are ruled out.
    """

    def verify(self, *, signals: StepSignals) -> DefectVerification:
        """
        Returns CONFIRMED for a well-corroborated dead tap, else NEEDS_REVIEW.

        A dead tap is only trusted when the tap was confidently grounded on a real
        target, the foreground activity did not change (no swallowed transition), and
        the tap landed on an interactive element. An unjudgeable hit (UNKNOWN) does
        not downgrade on its own, so screens without a usable hierarchy are not punished.
        """

        grounded = signals.grounding is not None and signals.grounding.is_corroborated
        confident = signals.confidence >= DEAD_TAP_MIN_CONFIDENCE
        stayed = signals.post_activity is None or signals.post_activity == signals.activity
        not_missed = signals.target_hit is not HitOutcome.MISS

        if grounded and confident and stayed and not_missed:
            return DefectVerification.CONFIRMED
        return DefectVerification.NEEDS_REVIEW
