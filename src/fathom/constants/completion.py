from __future__ import annotations

from enum import StrEnum


class CompletionEvidence(StrEnum):
    """
    Evidence dimensions inspected by the completion service.
    """

    TASK_STATUS_MET = "TASK_STATUS_MET"
    OUTCOME_BLOCKED = "OUTCOME_BLOCKED"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    OUTCOME_EFFECTIVE = "OUTCOME_EFFECTIVE"
