from __future__ import annotations

from enum import IntEnum, StrEnum


class AdvanceKind(StrEnum):
    """
    Decision families the advancement policy can produce for one turn.
    """

    RETAIN = "RETAIN"
    ADVANCE = "ADVANCE"
    ESCALATE = "ESCALATE"
    UNSATISFIABLE = "UNSATISFIABLE"
    SATISFIED_PRIOR = "SATISFIED_PRIOR"


class ObservationPhase(StrEnum):
    """
    Whether a turn's observation was read before or after the action was dispatched.
    """

    PRE_DISPATCH = "PRE_DISPATCH"
    POST_DISPATCH = "POST_DISPATCH"


class AdvanceThreshold(IntEnum):
    """
    Bounded-retention limits consumed by the advancement policy.
    """

    RETAIN_ESCALATION = 3
