from __future__ import annotations

from typing import Optional

from fathom.constants import ActionType
from fathom.schemas.actions import Action
from fathom.schemas.state import LoopDetector


class RecoveryActionLadder:
    """
    Picks the next mechanical recovery action when the agent is stuck in a loop.
    """

    BACK_RATIONALE = "Loop detected (screen repeating). Forcing BACK to break context."
    SCROLL_RATIONALE = "Loop detected (screen repeating). Forcing SCROLL to reveal new state."
    HOME_RATIONALE = "Loop detected (screen repeating). Forcing HOME to reset agent."

    def next(self, *, detector: LoopDetector) -> Optional[Action]:
        """
        Return the next recovery action when the loop detector still has budget.
        """

        if not detector.can_recover():
            return None

        attempt = detector.record_recovery_attempt()
        if attempt == 1:
            return Action(
                confidence=0.9,
                target="system: back",
                action_type=ActionType.BACK,
                rationale=self.BACK_RATIONALE,
            )

        if attempt == 2:
            return Action(
                confidence=0.8,
                target="system: scroll",
                action_type=ActionType.SCROLL,
                rationale=self.SCROLL_RATIONALE,
            )

        return Action(
            confidence=0.7,
            target="system: home",
            action_type=ActionType.HOME,
            rationale=self.HOME_RATIONALE,
        )
