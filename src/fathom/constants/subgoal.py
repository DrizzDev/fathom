from __future__ import annotations

from typing import Final

# Default per-sub-goal step budget. The agent gets this many ANALYZE -> EXECUTE
# -> RECORD cycles per sub-goal before the recovery coordinator is dispatched
# with SUBGOAL_BUDGET_EXCEEDED. Hard cap regardless of whether the loop detector
# or no-progress classifier has fired.
DEFAULT_SUB_GOAL_MAX_STEPS: Final[int] = 8
