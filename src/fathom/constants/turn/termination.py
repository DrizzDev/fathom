from __future__ import annotations

from enum import StrEnum


class TerminationStatus(StrEnum):
    """
    Honest terminal status of one run, resolved from the outcome and completion reason.
    """

    FAILED = "FAILED"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"
    NEEDS_INPUT = "NEEDS_INPUT"
    UNSATISFIABLE = "UNSATISFIABLE"
