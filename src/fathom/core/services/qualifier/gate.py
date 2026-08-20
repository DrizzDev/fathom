from __future__ import annotations

from fathom.constants.qualification import QualificationLabel, RationaleCategory
from fathom.schemas.configuration import QualifierConfiguration
from fathom.schemas.qualification import QualificationVerdict


class QualificationGatePolicy:
    """
    Binary gate: block iff the verdict is NOT_EXECUTABLE.

    Specificity and target-grounding are the strategy's responsibility. The gate
    only enforces "this is a mobile UI automation request" / "this is not."
    """

    def __init__(self, *, configuration: QualifierConfiguration) -> None:
        """
        Accept the qualifier configuration for symmetry with caller wiring.

        Unused at decision time — the gate has no threshold; kept as a parameter so callers holding a
        config need not special-case construction.
        """

        _ = configuration

    def should_block(self, *, verdict: QualificationVerdict) -> bool:
        """
        Block rules:
          - QUALIFIER_ERROR always passes (fail-open on any qualifier failure).
          - NOT_EXECUTABLE always blocks.
          - Everything else (EXECUTABLE) passes.
        """

        if verdict.rationale.category == RationaleCategory.QUALIFIER_ERROR:
            return False

        return verdict.label == QualificationLabel.NOT_EXECUTABLE
