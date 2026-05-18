from __future__ import annotations

from typing import List, Optional

from fathom.schemas.subgoal import SubGoal, SubGoalKind, SubGoalStatus
from fathom.schemas.tasks import (
    ExecutionTask,
    ExecutionTaskState,
    TaskAttemptState,
    TaskKind,
)


class ExecutionTaskAdapter:
    """
    Converts legacy sub-goals into execution tasks during migration.
    """

    def from_sub_goals(
        self,
        *,
        sub_goals: List[SubGoal],
        kinds: Optional[List[SubGoalKind]] = None,
    ) -> List[ExecutionTask]:
        """
        Convert legacy sub-goals into execution tasks, optionally with explicit kinds.
        """

        if kinds is None:
            return [self.from_sub_goal(sub_goal=sub_goal) for sub_goal in sub_goals]

        return [
            self.from_sub_goal(sub_goal=sub_goal, kind=kind)
            for sub_goal, kind in zip(sub_goals, kinds, strict=False)
        ]

    def from_sub_goal(
        self,
        *,
        sub_goal: SubGoal,
        kind: Optional[SubGoalKind] = None,
    ) -> ExecutionTask:
        """
        Convert one legacy sub-goal into an execution task.
        """

        return ExecutionTask(
            kind=self.__kind(kind=kind),
            objective=sub_goal.description,
            identifier=f"task:{sub_goal.index}",
            state=self.__state(status=sub_goal.status),
            criterion=sub_goal.criterion or sub_goal.description,
            attempts=TaskAttemptState(count=0, limit=sub_goal.max_steps),
        )

    def __state(self, *, status: SubGoalStatus) -> ExecutionTaskState:
        """
        Convert legacy sub-goal state into execution task state.
        """

        if status == SubGoalStatus.COMPLETE:
            return ExecutionTaskState.SUCCEEDED

        if status == SubGoalStatus.FAILED:
            return ExecutionTaskState.FAILED

        if status == SubGoalStatus.IN_PROGRESS:
            return ExecutionTaskState.ACTIVE

        return ExecutionTaskState.PENDING

    @staticmethod
    def __kind(*, kind: Optional[SubGoalKind]) -> TaskKind:
        """
        Map legacy SubGoalKind onto the runtime TaskKind enum.
        """

        if kind == SubGoalKind.VALIDATION:
            return TaskKind.VALIDATION

        return TaskKind.ACTION
