from __future__ import annotations

from typing import Optional

from fathom.constants import ActionType
from fathom.schemas.actions import Action


class ActionFixtures:
    """
    Factory for :class:`Action` instances used across planner / reasoner tests.
    """

    @classmethod
    def make(
        cls,
        *,
        target: str = "target",
        action_type: ActionType = ActionType.TAP,
        rationale: str = "test rationale",
        confidence: float = 0.9,
        natural_language_target: Optional[str] = None,
        text: Optional[str] = None,
    ) -> Action:
        """
        Build an :class:`Action` whose defaults satisfy planner/reasoner tests.
        """

        return Action(
            target=target,
            action_type=action_type,
            rationale=rationale,
            confidence=confidence,
            natural_language_target=natural_language_target,
            text=text,
        )

    @classmethod
    def tap(
        cls,
        *,
        target: str = "Continue",
        rationale: str = "tap continue",
        confidence: float = 0.9,
    ) -> Action:
        """
        Convenience constructor for the common TAP action shape.
        """

        return cls.make(
            target=target,
            action_type=ActionType.TAP,
            rationale=rationale,
            confidence=confidence,
        )
