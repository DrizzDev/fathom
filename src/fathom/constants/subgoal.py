from __future__ import annotations

from enum import StrEnum
from typing import Final

# Default per-sub-goal step budget. The agent gets this many ANALYZE -> EXECUTE
# -> RECORD cycles per sub-goal before the planner must choose another path.
DEFAULT_SUB_GOAL_MAX_STEPS: Final[int] = 8
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
