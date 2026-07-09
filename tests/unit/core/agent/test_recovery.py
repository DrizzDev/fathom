from __future__ import annotations

import unittest

from fathom.constants import ActionType
from fathom.constants.recovery import AUTONOMOUS_RECOVERY_ACTIVE_KINDS
from fathom.core.agent.recovery import RecoveryGate
from fathom.schemas.effect import ActionEffectStatus
from fathom.schemas.loop import LoopEvidence, LoopReason, LoopTurn
from fathom.schemas.recovery import RecoveryDecisionKind, RecoveryReason
from fathom.schemas.vision import ActionKind, ActionKindResolver


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

    def test_swipe_no_progress_blocks_blind_back_recovery(self) -> None:
        """
        Every swipe/scroll action type with NO_PROGRESS must classify as active and force REPLAN end-to-end.
        Exercises the real ActionType -> ActionKindResolver chain so a future kind regression cannot pass silently.
        """

        for action_type in (
            ActionType.SWIPE_LEFT,
            ActionType.SWIPE_RIGHT,
            ActionType.SWIPE_UP,
            ActionType.SWIPE_DOWN,
            ActionType.SCROLL,
        ):
            with self.subTest(action_type=action_type):
                evidence = self.__evidence(
                    turn=LoopTurn(
                        action_kind=ActionKindResolver.resolve(action_type=action_type),
                        action_type=action_type.value,
                        effect_status=ActionEffectStatus.NO_PROGRESS,
                    )
                )

                decision = RecoveryGate(active_kinds=AUTONOMOUS_RECOVERY_ACTIVE_KINDS).decide(
                    evidence=evidence,
                )

                self.assertEqual(decision.kind, RecoveryDecisionKind.REPLAN)
                self.assertEqual(decision.reason, RecoveryReason.ACTIVE_NO_PROGRESS)

    def test_active_progress_or_uncertain_does_not_force_replan(self) -> None:
        """
        Only NO_PROGRESS on an active turn triggers REPLAN; PROGRESS and UNCERTAIN remain eligible for the ladder.
        """

        for status in (ActionEffectStatus.PROGRESS, ActionEffectStatus.UNCERTAIN):
            with self.subTest(status=status):
                evidence = self.__evidence(
                    turn=LoopTurn(
                        action_kind=ActionKindResolver.resolve(action_type=ActionType.SWIPE_LEFT),
                        action_type=ActionType.SWIPE_LEFT.value,
                        effect_status=status,
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
