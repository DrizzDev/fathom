from __future__ import annotations

from typing import Any, Dict

from fathom.constants.runtime import DEFAULT_REALIGNMENT_BUDGET


class RealignmentTracker:
    """
    Tracks HITL realignment intervention usage against a per-run budget.
    """

    def __init__(self, *, budget: int = DEFAULT_REALIGNMENT_BUDGET) -> None:
        """
        Initialize the tracker with the per-run intervention budget.
        """

        self.__budget = budget
        self.__count: int = 0

    @property
    def budget(self) -> int:
        """
        Return the configured per-run realignment budget.
        """

        return self.__budget

    @property
    def count(self) -> int:
        """
        Return the number of realignment interventions used so far this run.
        """

        return self.__count

    def record(self) -> None:
        """
        Increment the realignment counter by one.
        """

        self.__count += 1

    def exhausted(self) -> bool:
        """
        Return whether the realignment budget has been exhausted.
        """

        return self.__count >= self.__budget

    def to_state(self) -> Dict[str, Any]:
        """
        Serialize the realignment counters for checkpoint persistence.
        """

        return {
            "budget": self.__budget,
            "count": self.__count,
        }

    def load_state(self, *, state: Dict[str, Any]) -> None:
        """
        Replace the in-memory counters with a checkpoint payload.
        """

        if isinstance(budget := state.get("budget"), int) and budget > 0:
            self.__budget = budget

        if isinstance(count := state.get("count", 0), int) and count >= 0:
            self.__count = count
