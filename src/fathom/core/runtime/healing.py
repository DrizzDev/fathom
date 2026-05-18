from __future__ import annotations

from collections import defaultdict
from typing import Any, DefaultDict, Dict


class HealingUsage:
    """
    Tracks per-task and per-run healing-decision usage for budget enforcement.
    """

    def __init__(self) -> None:
        """
        Initialize empty healing-usage counters.
        """

        self.__per_run: int = 0
        self.__per_task: DefaultDict[str, int] = defaultdict(int)

    def record(self, *, task_id: str) -> None:
        """
        Increment counters for one healing decision against the given task.
        """

        self.__per_run += 1
        self.__per_task[task_id] += 1

    def task_count(self, *, task_id: str) -> int:
        """
        Return the healing-decision count for one task.
        """

        return self.__per_task[task_id]

    def run_count(self) -> int:
        """
        Return the healing-decision count for the current run.
        """

        return self.__per_run

    def reset_task(self, *, task_id: str) -> None:
        """
        Clear the per-task healing counter on task replan or advance.
        """

        self.__per_task.pop(task_id, None)

    def to_state(self) -> Dict[str, Any]:
        """
        Serialize the healing-usage counters for checkpoint persistence.
        """

        return {
            "per_run": self.__per_run,
            "per_task": dict(self.__per_task),
        }

    def load_state(self, *, state: Dict[str, Any]) -> None:
        """
        Replace the in-memory counters with a checkpoint payload; require both keys.
        """

        if "per_run" not in state or "per_task" not in state:
            raise ValueError(
                "HealingUsage.load_state requires both 'per_run' and 'per_task' keys; "
                f"received {sorted(state.keys())!r}.",
            )

        per_run = state["per_run"]
        per_task_raw = state["per_task"]
        if not isinstance(per_run, int) or per_run < 0:
            raise ValueError("HealingUsage.load_state expects 'per_run' to be a non-negative int.")
        if not isinstance(per_task_raw, dict):
            raise ValueError("HealingUsage.load_state expects 'per_task' to be a dict.")

        self.__per_run = per_run
        self.__per_task.clear()
        for task_id, count in per_task_raw.items():
            if isinstance(task_id, str) and isinstance(count, int) and count >= 0:
                self.__per_task[task_id] = count
