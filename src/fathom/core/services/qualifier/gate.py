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

        The configuration is intentionally unused at decision time — the gate
        has no threshold. Kept as a constructor parameter so callers don't
        have to special-case construction when they already hold a config.
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
