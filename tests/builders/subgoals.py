from __future__ import annotations

from typing import Optional

from fathom.schemas.subgoal import GoalState, SubGoal
from fathom.schemas.success import ObservationRequirement, ObservedSuccess, Success


class SubGoalFixtures:
    """
    Factory for :class:`SubGoal` / :class:`GoalState` instances used across planner and completion tests.
    """

    @classmethod
    def make(
        cls,
        *,
        description: str = "active sub-goal",
        index: int = 0,
        success: Optional[Success] = None,
    ) -> SubGoal:
        """
        Build a :class:`SubGoal`; success defaults to an observed objective mirroring the description.
        """

        return SubGoal(
            index=index,
            objective=description,
            success=success
            if success is not None
            else ObservedSuccess(observation=ObservationRequirement(assertion=description)),
        )

    @classmethod
    def state(
        cls,
        *,
        description: str = "active sub-goal",
        index: int = 0,
        success: Optional[Success] = None,
    ) -> GoalState:
        """
        Build a :class:`GoalState` wrapping a fresh sub-goal with default progress.
        """

        return GoalState(goal=cls.make(description=description, index=index, success=success))
