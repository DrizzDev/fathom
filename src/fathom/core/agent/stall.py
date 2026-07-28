from __future__ import annotations

from typing import Sequence

from fathom.constants.turn.stall import StallState, StallThreshold
from fathom.schemas.effect import ActionEffect, ActionEffectStatus
from fathom.schemas.stall import StallSignal


class StallPolicy:
    """
    Classifies momentum from the typed effect stream; UNCERTAIN effects count toward the stall.
    """

    def __init__(self, *, limit: int = StallThreshold.STREAK) -> None:
        """
        Bind the trailing non-progress streak limit.
        """

        self.__limit = limit

    def assess(self, *, effects: Sequence[ActionEffect]) -> StallSignal:
        """
        Return the momentum reading for the trailing effect stream.
        """

        streak = 0
        for effect in reversed(effects):
            if effect.status is ActionEffectStatus.PROGRESS:
                break
            streak += 1

        if streak == 0:
            return StallSignal(state=StallState.FLOWING, streak=0)

        if streak >= self.__limit:
            return StallSignal(state=StallState.STALLED, streak=streak)

        return StallSignal(state=StallState.UNCERTAIN, streak=streak)
