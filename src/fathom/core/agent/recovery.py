from __future__ import annotations

from typing import FrozenSet

from fathom.schemas.effect import ActionEffectStatus
from fathom.schemas.loop import LoopEvidence
from fathom.schemas.recovery import RecoveryDecision, RecoveryDecisionKind, RecoveryReason
from fathom.schemas.vision import ActionKind


class RecoveryGate:
    """
    Decides whether autonomous mechanical recovery is safe for loop evidence.
    """

    def __init__(self, *, active_kinds: FrozenSet[ActionKind]) -> None:
        """
        Bind the action kinds considered active device work.
        """

        self.__active_kinds = active_kinds

    def decide(self, *, evidence: LoopEvidence) -> RecoveryDecision:
        """
        Return whether blind mechanical recovery may run for the evidence.
        """

        if self.__has_active_no_progress(evidence=evidence):
            return RecoveryDecision(
                kind=RecoveryDecisionKind.REPLAN,
                reason=RecoveryReason.ACTIVE_NO_PROGRESS,
            )

        return RecoveryDecision(
            reason=RecoveryReason.SAFE,
            kind=RecoveryDecisionKind.ALLOW,
        )

    def __has_active_no_progress(self, *, evidence: LoopEvidence) -> bool:
        """
        Return whether active device work failed without observable progress.
        """

        return any(
            turn.effect_status is ActionEffectStatus.NO_PROGRESS
            and turn.action_kind in self.__active_kinds
            for turn in evidence.since_progress
        )
