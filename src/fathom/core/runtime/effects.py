from __future__ import annotations

from collections import deque
from typing import Deque, List, Optional

from fathom.constants.screen import ACTION_EFFECT_TRAJECTORY_WINDOW
from fathom.schemas.effect import ActionEffect, ActionEffectStatus
from fathom.schemas.outcomes import ActionOutcome, OutcomeStatus


class EffectHistory:
    """
    Maintains bounded action-effect and action-outcome history.
    """

    def __init__(self, *, window: int = ACTION_EFFECT_TRAJECTORY_WINDOW) -> None:
        """
        Initialize bounded history windows.
        """

        self.__effects: Deque[ActionEffect] = deque(maxlen=window)
        self.__outcomes: Deque[ActionOutcome] = deque(maxlen=window)

    def record_effect(self, *, effect: ActionEffect) -> None:
        """
        Append one deterministic action-effect summary.
        """

        self.__effects.append(effect)

    def record_outcome(self, *, outcome: ActionOutcome) -> None:
        """
        Append one action-aware outcome.
        """

        self.__outcomes.append(outcome)

    def load_effects(self, *, effects: List[ActionEffect]) -> None:
        """
        Replace effect history with a restored checkpoint window.
        """

        self.__effects.clear()
        self.__effects.extend(effects)

    def clear(self) -> None:
        """
        Clear effect and outcome history.
        """

        self.__effects.clear()
        self.__outcomes.clear()

    def recent_effects(self) -> List[ActionEffect]:
        """
        Return recent action effects oldest first.
        """

        return list(self.__effects)

    def recent_outcomes(self) -> List[ActionOutcome]:
        """
        Return recent action outcomes oldest first.
        """

        return list(self.__outcomes)

    def last_effect(self) -> Optional[ActionEffect]:
        """
        Return the latest action effect when available.
        """

        if not self.__effects:
            return None

        return self.__effects[-1]

    def last_outcome(self) -> Optional[ActionOutcome]:
        """
        Return the latest action outcome when available.
        """

        if not self.__outcomes:
            return None

        return self.__outcomes[-1]

    def consecutive_no_progress(self) -> int:
        """
        Count trailing no-progress action effects.
        """

        count = 0
        for effect in reversed(self.__effects):
            if effect.status != ActionEffectStatus.NO_PROGRESS:
                break
            count += 1

        return count

    def consecutive_no_effect(self) -> int:
        """
        Count trailing no-effect action outcomes.
        """

        count = 0

        for outcome in reversed(self.__outcomes):
            if outcome.status != OutcomeStatus.NO_EFFECT:
                break

            count += 1

        return count
