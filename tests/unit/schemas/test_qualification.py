from __future__ import annotations

import unittest

from pydantic import ValidationError

from fathom.constants.qualification import QualificationLabel, RationaleCategory
from fathom.schemas.qualification import QualificationVerdict, Rationale


class QualificationVerdictTest(unittest.TestCase):
    """
    QualificationVerdict is a value object; it carries data only, no decision methods.
    """

    @staticmethod
    def _verdict(*, label: QualificationLabel, confidence: float) -> QualificationVerdict:
        """
        Construct a verdict with a uniform rationale shape.
        """

        return QualificationVerdict(
            label=label,
            confidence=confidence,
            rationale=Rationale(category=RationaleCategory.UI_TASK, reasoning="test"),
        )

    def test_confidence_bounds_are_validated(self) -> None:
        """
        Confidence outside the 0.0-1.0 range must raise during construction.
        """

        with self.assertRaises(ValidationError):
            self._verdict(label=QualificationLabel.EXECUTABLE, confidence=1.5)

    def test_rationale_is_nested_pydantic_model(self) -> None:
        """
        Rationale must be exposed as a nested Pydantic model, not a flat field.
        """

        verdict = self._verdict(label=QualificationLabel.EXECUTABLE, confidence=0.95)
        self.assertIsInstance(verdict.rationale, Rationale)
        self.assertEqual(verdict.rationale.category, RationaleCategory.UI_TASK)

    def test_verdict_is_frozen(self) -> None:
        """
        Verdict must be immutable so downstream consumers cannot mutate the decision.
        """

        verdict = self._verdict(label=QualificationLabel.EXECUTABLE, confidence=0.9)
        with self.assertRaises(ValidationError):
            verdict.confidence = 0.5  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
