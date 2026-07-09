from __future__ import annotations

from enum import IntEnum, StrEnum
from typing import Final, FrozenSet

DURABLE_OUTCOME_TERMS: Final[FrozenSet[str]] = frozenset(
    {
        "add",
        "added",
        "adding",
        "capture",
        "captured",
        "save",
        "saved",
        "store",
        "stored",
    }
)


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

    MISSING_CLAIM = "MISSING_CLAIM"
    MISSING_CAPTURE = "MISSING_CAPTURE"
    MISSING_DISPATCH = "MISSING_DISPATCH"
    MISSING_VALIDATION = "MISSING_VALIDATION"
    MISSING_JUSTIFICATION = "MISSING_JUSTIFICATION"
    MISSING_CAPTURE_REQUEST = "MISSING_CAPTURE_REQUEST"
    MISSING_OUTCOME_EVIDENCE = "MISSING_OUTCOME_EVIDENCE"
    MISSING_SCREEN_EVOLUTION = "MISSING_SCREEN_EVOLUTION"

    CAPTURE_FAILED = "CAPTURE_FAILED"
    EMPTY_CAPTURE_VALUE = "EMPTY_CAPTURE_VALUE"
    STEP_EXECUTION_FAILED = "STEP_EXECUTION_FAILED"


class GateOutcome(StrEnum):
    """
    Per-turn completion-gate decision.
    """

    FAIL = "FAIL"
    RETAIN = "RETAIN"
    ADVANCE = "ADVANCE"


class AdvanceReason(StrEnum):
    """
    Diagnostic code explaining which gate branch ratified an ADVANCE this turn.
    """

    STRICT_PATH = "STRICT_PATH"
    VALIDATION_ACTION = "VALIDATION_ACTION"
    VALIDATION_IMPLICIT_COMPLETION = "VALIDATION_IMPLICIT_COMPLETION"
