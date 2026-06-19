from __future__ import annotations

from typing import Optional

from fathom.schemas.subgoal import SubGoal, SubGoalKind


class SubGoalFixtures:
    """
    Factory for :class:`SubGoal` instances used across planner / completion tests.
    """

    @classmethod
    def make(
        cls,
        *,
        description: str = "active sub-goal",
        index: int = 0,
        max_steps: int = 10,
        kind: Optional[SubGoalKind] = None,
    ) -> SubGoal:
        """
        Build a :class:`SubGoal` with sane defaults; ``kind`` is only set when supplied.
        """

        if kind is None:
            return SubGoal(description=description, index=index, max_steps=max_steps)

        return SubGoal(
            description=description,
            index=index,
            max_steps=max_steps,
            kind=kind,
        )
