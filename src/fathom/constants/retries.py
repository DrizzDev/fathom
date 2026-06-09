from __future__ import annotations

from enum import StrEnum
from typing import Final

MIN_RETRY_CAP: Final[int] = 1
DEFAULT_PLANNER_RETRY_LIMIT: Final[int] = 5

PLANNER_RETRY_CONSUMED: Final[str] = "PLANNER_RETRY_CONSUMED"
PLANNER_RETRY_EXHAUSTED: Final[str] = "PLANNER_RETRY_EXHAUSTED"
PLANNER_RETRY_METADATA_INVALID: Final[str] = "PLANNER_RETRY_METADATA_INVALID"


class RetryKind(StrEnum):
    """
    Classifies which should_retry=True branch produced the planner return.
    """

    LLM_FEEDBACK = "LLM_FEEDBACK"
    SILENT_REJECTION = "SILENT_REJECTION"
    ESCALATION_DEFERRED = "ESCALATION_DEFERRED"


class RetryBranch(StrEnum):
    """
    Stable branch identifiers stamped into PlanResult.metadata so analyze can route the right budget.
    """

    UNKNOWN = "UNKNOWN"
    SHOULD_AVOID_ACTION = "SHOULD_AVOID_ACTION"
    ESCALATION_DEFERRED = "ESCALATION_DEFERRED"
    CURRENT_SCREEN_REPEAT = "CURRENT_SCREEN_REPEAT"
    IS_ACTION_REPEATING_ON_SCREEN = "IS_ACTION_REPEATING_ON_SCREEN"


class RetryMetadataField(StrEnum):
    """
    Stable keys the planner writes into PlanResult.metadata.
    """

    KIND = "RETRY_KIND"
    BRANCH = "RETRY_BRANCH"
    BLOCK_REASON = "BLOCK_REASON"
    BLOCKED_ACTION = "BLOCKED_ACTION"
