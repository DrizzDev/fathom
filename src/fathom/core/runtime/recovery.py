from __future__ import annotations

from collections import defaultdict
from typing import DefaultDict, Tuple

from fathom.core.recovery.types import RecoveryTrigger


class RecoveryRuntimeState:
    """
    Tracks scoped recovery attempts without owning recovery strategy logic.
    """

    def __init__(self) -> None:
        """
        Initialize empty recovery counters.
        """

        self.__counters: DefaultDict[Tuple[int, RecoveryTrigger], int] = defaultdict(int)

    def record(self, *, scope: int, trigger: RecoveryTrigger) -> int:
        """
        Increment and return the recovery count for a scope and trigger.
        """

        key = (scope, trigger)

        self.__counters[key] += 1
        return self.__counters[key]

    def count(self, *, scope: int, trigger: RecoveryTrigger) -> int:
        """
        Return the recovery count for a scope and trigger.
        """

        return self.__counters[(scope, trigger)]

    def reset(self, *, scope: int) -> None:
        """
        Clear all recovery counts for a scope.
        """

        for key in list(self.__counters):
            if key[0] == scope:
                del self.__counters[key]
