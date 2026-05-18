from __future__ import annotations

from typing import List, Optional, Tuple

from fathom.schemas.tasks import ExecutionTask, ExecutionTaskState, TaskAttemptState


class TaskRuntimeState:
    """
    Owns execution-task progress and attempt accounting.
    """

    def __init__(self, *, tasks: Optional[List[ExecutionTask]] = None) -> None:
        """
        Initialize task state.
        """

        self.__index = 0
        self.__tasks = list(tasks or [])

    def load(self, *, tasks: List[ExecutionTask]) -> None:
        """
        Replace the task plan and reset the active index.
        """

        self.__index = 0
        self.__tasks = list(tasks)

    def active(self) -> Optional[ExecutionTask]:
        """
        Return the active execution task when present.
        """

        if not self.__tasks or self.__index >= len(self.__tasks):
            return None

        return self.__tasks[self.__index]

    def all(self) -> List[ExecutionTask]:
        """
        Return all execution tasks.
        """

        return list(self.__tasks)

    def progress(self) -> Tuple[int, int]:
        """
        Return active index and task count.
        """

        return self.__index, len(self.__tasks)

    def record_attempt(self) -> None:
        """
        Increment the active task attempt count.
        """

        task = self.active()
        if task is None:
            return

        self.__tasks[self.__index] = task.model_copy(
            update={
                "attempts": TaskAttemptState(
                    limit=task.attempts.limit,
                    count=task.attempts.count + 1,
                )
            }
        )

    def mark(self, *, state: ExecutionTaskState) -> None:
        """
        Update the active task state.
        """

        task = self.active()
        if task is None:
            return

        self.__tasks[self.__index] = task.model_copy(update={"state": state})

    def advance(self) -> bool:
        """
        Advance to the next task and return whether one exists.
        """

        if self.__index + 1 >= len(self.__tasks):
            return False

        self.__index += 1
        return True

    def over_budget(self) -> bool:
        """
        Return whether the active task has exhausted its budget.
        """

        task = self.active()
        return bool(task and task.over_budget)
