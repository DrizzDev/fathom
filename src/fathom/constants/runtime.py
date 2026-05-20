from __future__ import annotations

from typing import Final

# Time-based budgets (milliseconds)
DEFAULT_LOCALIZATION_BUDGET: Final[int] = 5000
DEFAULT_OCR_PERCEPTION_BUDGET: Final[int] = 5000
DEFAULT_LOCAL_PERCEPTION_BUDGET: Final[int] = 5000


# Non-time runtime knobs
# Confidence floor for accepting a localization match. Dimensionless ratio in [0, 1]; not a timeout.
DEFAULT_LOCALIZATION_CONFIDENCE_THRESHOLD: Final[float] = 0.72

DEFAULT_HEALING_RUN_BUDGET: Final[int] = 5
DEFAULT_HEALING_TASK_BUDGET: Final[int] = 2
DEFAULT_PAID_LOCALIZATION_ATTEMPT_BUDGET: Final[int] = 0


DEFAULT_LOOP_WINDOW: Final[int] = 10
DEFAULT_LOOP_THRESHOLD: Final[int] = 3

# Minimum identical consecutive (action, NO_PROGRESS) pairs that count as an "inert repetition
DEFAULT_INERT_REPETITION_THRESHOLD: Final[int] = 2

DEFAULT_MAX_STEPS: Final[int] = 20
DEFAULT_CONTEXT_WINDOW: Final[int] = 10
DEFAULT_REALIGNMENT_BUDGET: Final[int] = 3

# Consecutive ANALYZE turns whose ``is_complete=True`` verdict the router
# may defer because sub-goals are still open. Beyond this budget the planner
# has stably claimed completion for the same screen state; honouring the
# claim avoids a ground-loop and lets VERIFY adjudicate the final outcome.
DEFAULT_COMPLETE_DEFERRAL_BUDGET: Final[int] = 2
