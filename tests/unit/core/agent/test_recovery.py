from __future__ import annotations

import unittest

from fathom.constants.recovery import AUTONOMOUS_RECOVERY_ACTIVE_KINDS
from fathom.core.agent.recovery import RecoveryGate
from fathom.schemas.effect import ActionEffectStatus
from fathom.schemas.loop import LoopEvidence, LoopReason, LoopTurn
from fathom.schemas.recovery import RecoveryDecisionKind, RecoveryReason
from fathom.schemas.vision import ActionKind


class RecoveryGateTest(unittest.TestCase):
    """
    Pins autonomous recovery eligibility decisions.
    """

    def test_active_no_progress_requests_replan(self) -> None:
        """
        Active no-progress evidence must block blind mechanical recovery.
        """

        evidence = self.__evidence(
            turn=LoopTurn(
                action_kind=ActionKind.NAVIGATION,
                action_type="tap",
                effect_status=ActionEffectStatus.NO_PROGRESS,
            )
        )

        decision = RecoveryGate(active_kinds=AUTONOMOUS_RECOVERY_ACTIVE_KINDS).decide(
            evidence=evidence,
        )

        self.assertEqual(decision.kind, RecoveryDecisionKind.REPLAN)
        self.assertEqual(decision.reason, RecoveryReason.ACTIVE_NO_PROGRESS)

    def test_passive_no_progress_allows_mechanical_recovery(self) -> None:
        """
        Passive no-progress evidence remains eligible for ladder recovery.
        """

        evidence = self.__evidence(
            turn=LoopTurn(
                action_kind=ActionKind.VALIDATION,
                action_type="validate",
                effect_status=ActionEffectStatus.NO_PROGRESS,
            )
        )

        decision = RecoveryGate(active_kinds=AUTONOMOUS_RECOVERY_ACTIVE_KINDS).decide(
            evidence=evidence,
        )

        self.assertEqual(decision.kind, RecoveryDecisionKind.ALLOW)
        self.assertEqual(decision.reason, RecoveryReason.SAFE)

    @staticmethod
    def __evidence(*, turn: LoopTurn) -> LoopEvidence:
        """
        Build a loop evidence snapshot from one contributing turn.
        """

        return LoopEvidence(
            stuck=True,
            reason=LoopReason.INERT_REPETITION,
            recent=(turn,),
            since_progress=(turn,),
        )
