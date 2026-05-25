from __future__ import annotations

from typing import Optional

from fathom.constants import ActionType
from fathom.constants.state import (
    LOOP_BACK_CONFIDENCE,
    LOOP_BACK_RATIONALE,
    LOOP_HOME_CONFIDENCE,
    LOOP_HOME_RATIONALE,
    LOOP_SCROLL_ACTION_TYPES,
    LOOP_SCROLL_CONFIDENCE,
    LOOP_SCROLL_RATIONALE,
)
from fathom.schemas.actions import Action
from fathom.schemas.state import LoopDetector


class LoopActionLadder:
    """
    Picks the next built-in loop-breaking action when the legacy loop detector fires.
    """

    def next(self, *, detector: LoopDetector) -> Optional[Action]:
        """
        Return the next loop-breaking action when the detector still has budget.
        """

        if not detector.can_recover():
            return None

        attempt = detector.record_recovery_attempt()
        if attempt == 1:
            return Action(
                target="system: back",
                action_type=ActionType.BACK,
                rationale=LOOP_BACK_RATIONALE,
                confidence=LOOP_BACK_CONFIDENCE,
            )

        if attempt == 2 and not self.__is_scroll_loop(detector=detector):
            return Action(
                target="system: scroll",
                action_type=ActionType.SCROLL,
                rationale=LOOP_SCROLL_RATIONALE,
                confidence=LOOP_SCROLL_CONFIDENCE,
            )

        return Action(
            target="system: home",
            action_type=ActionType.HOME,
            rationale=LOOP_HOME_RATIONALE,
            confidence=LOOP_HOME_CONFIDENCE,
        )

    @staticmethod
    def __is_scroll_loop(*, detector: LoopDetector) -> bool:
        """
        Return whether the stuck evidence already came from a scroll-like action.
        """

        action_type = detector.last_action_type
        return action_type in LOOP_SCROLL_ACTION_TYPES if action_type else False
