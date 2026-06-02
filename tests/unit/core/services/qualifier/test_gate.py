from __future__ import annotations

import unittest

from fathom.constants.qualification import QualificationLabel, RationaleCategory
from fathom.core.services.qualifier.gate import QualificationGatePolicy
from fathom.schemas.configuration import QualifierConfiguration
from fathom.schemas.qualification import QualificationVerdict, Rationale


class QualificationGatePolicyTest(unittest.TestCase):
    """
    Binary gate: block iff label is NOT_EXECUTABLE. QUALIFIER_ERROR fails open.
    """

    @staticmethod
    def _verdict(
        *,
        label: QualificationLabel,
        confidence: float = 0.95,
        category: RationaleCategory = RationaleCategory.UI_TASK,
    ) -> QualificationVerdict:
        """
        Construct a verdict with a uniform shape.
        """

        return QualificationVerdict(
            label=label,
            confidence=confidence,
            rationale=Rationale(category=category, reasoning="test"),
        )

    def setUp(self) -> None:
        """
        Provide the default qualifier configuration for every test.
        """

        self.__policy = QualificationGatePolicy(configuration=QualifierConfiguration())

    def test_executable_passes_at_any_confidence(self) -> None:
        """
        EXECUTABLE must pass regardless of confidence (binary gate, no threshold).
        """

        for confidence in (0.1, 0.5, 0.85, 0.99):
            with self.subTest(confidence=confidence):
                verdict = self._verdict(label=QualificationLabel.EXECUTABLE, confidence=confidence)
                self.assertFalse(self.__policy.should_block(verdict=verdict))

    def test_not_executable_blocks_at_any_confidence(self) -> None:
        """
        NOT_EXECUTABLE always blocks; the binary gate ignores confidence.
        """

        for confidence in (0.1, 0.5, 0.85, 0.99):
            with self.subTest(confidence=confidence):
                verdict = self._verdict(
                    label=QualificationLabel.NOT_EXECUTABLE,
                    confidence=confidence,
                    category=RationaleCategory.GIBBERISH,
                )
                self.assertTrue(self.__policy.should_block(verdict=verdict))

    def test_qualifier_error_never_blocks(self) -> None:
        """
        Fail-open: any verdict marked QUALIFIER_ERROR passes regardless of label
        or confidence; the gate never blocks on infrastructure failures.
        """

        verdict = self._verdict(
            label=QualificationLabel.NOT_EXECUTABLE,
            confidence=0.99,
            category=RationaleCategory.QUALIFIER_ERROR,
        )
        self.assertFalse(self.__policy.should_block(verdict=verdict))


if __name__ == "__main__":
    unittest.main()
