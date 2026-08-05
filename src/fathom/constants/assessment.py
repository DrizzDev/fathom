from __future__ import annotations

from enum import StrEnum


class VisualVerdict(StrEnum):
    """
    The model's judgement of whether the active visual requirement holds on the current screenshot.
    """

    SATISFIED = "SATISFIED"
    NOT_SATISFIED = "NOT_SATISFIED"
    UNCLEAR = "UNCLEAR"


class PhaseComparison(StrEnum):
    """
    Whether a shadow phase's candidate and live decision rest on equivalent evidence.
    """

    COMPARABLE = "COMPARABLE"
    INCOMPARABLE = "INCOMPARABLE"


class PhaseIncomparability(StrEnum):
    """
    Why a shadow phase's candidate and live decision cannot be compared.
    """

    EXECUTION_FAILED = "EXECUTION_FAILED"
    VISUAL_EVIDENCE_DEFERRED = "VISUAL_EVIDENCE_DEFERRED"
    EVIDENCE_SOURCE_DIFFERENT = "EVIDENCE_SOURCE_DIFFERENT"


class ShadowDivergenceKind(StrEnum):
    """
    A way the shadow visual assessment disagrees with live behavior or the goal's declared evidence source.
    """

    MISSING_ASSESSMENT = "MISSING_ASSESSMENT"
    WRONG_GOAL = "WRONG_GOAL"
    SATISFIED_WITH_ACTION = "SATISFIED_WITH_ACTION"
    PACKAGE_CONTRADICTION = "PACKAGE_CONTRADICTION"
    FALSE_POSITIVE = "FALSE_POSITIVE"
    FALSE_NEGATIVE = "FALSE_NEGATIVE"
    SCHEMA_FAILURE = "SCHEMA_FAILURE"
