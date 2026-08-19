from __future__ import annotations

from enum import StrEnum
from typing import Final

# Default per-sub-goal step budget. The agent gets this many ANALYZE -> EXECUTE
# -> RECORD cycles per sub-goal before the planner must choose another path.
# Temporarily raised 8 -> 12 so multi-action sub-goals (e.g. "add 3 items") fit; revisit with decomposition.
DEFAULT_SUB_GOAL_MAX_STEPS: Final[int] = 12
SCROLL_SUB_GOAL_MAX_STEPS: Final[int] = 25
TAP_SUB_GOAL_MAX_STEPS: Final[int] = 3
INPUT_SUB_GOAL_MAX_STEPS: Final[int] = 5
WAIT_SUB_GOAL_MAX_STEPS: Final[int] = 3
VALIDATE_SUB_GOAL_MAX_STEPS: Final[int] = 3


class TaskProof(StrEnum):
    """
    Proof requirement the decomposer declares for one task's completion.
    """

    DURABLE = "DURABLE"
    TRANSIENT = "TRANSIENT"
