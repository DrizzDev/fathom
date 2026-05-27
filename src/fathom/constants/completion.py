from __future__ import annotations

from enum import IntEnum, StrEnum


class VerifyEvidenceDimension(StrEnum):
    """
    Evidence dimensions inspected by the verify-node completion service.
    """

    TASK_STATUS_MET = "TASK_STATUS_MET"
    OUTCOME_BLOCKED = "OUTCOME_BLOCKED"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    OUTCOME_EFFECTIVE = "OUTCOME_EFFECTIVE"


class CompletionThreshold(IntEnum):
    """
    Required-signal counts for the completion gate per sub-goal kind.
    """

    ACTION_TOTAL = 3
    VALIDATION_WITHOUT_CLAIM = 2


class RetainReason(StrEnum):
    """
    Diagnostic code explaining why the completion gate retained the current sub-goal.
    """

    MISSING_CLAIM = "missing.claim"
    MISSING_DISPATCH = "missing.dispatch"
    MISSING_JUSTIFICATION = "missing.justification"
    MISSING_SCREEN_EVOLUTION = "missing.screen.evolution"


class GateOutcome(StrEnum):
    """
    Per-turn completion-gate decision.
    """

    FAIL = "fail"
    RETAIN = "retain"
    ADVANCE = "advance"
