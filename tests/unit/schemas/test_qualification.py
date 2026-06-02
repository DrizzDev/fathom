from __future__ import annotations

import unittest

from pydantic import ValidationError

from fathom.constants.qualification import QualificationLabel, RationaleCategory
from fathom.schemas.configuration import QualifierConfiguration
from fathom.schemas.qualification import QualificationVerdict, Rationale


class QualificationVerdictTest(unittest.TestCase):
    """
    Threshold rule must reject only on NOT_EXECUTABLE at or above the configured floor.
    """

    def setUp(self) -> None:
        """
        Capture the default qualifier configuration once per test.
        """

        self.__configuration = QualifierConfiguration()

    @staticmethod
    def _verdict(*, label: QualificationLabel, confidence: float) -> QualificationVerdict:
        """
        Construct a verdict with a uniform rationale shape for threshold checks.
        """

        return QualificationVerdict(
            label=label,
            confidence=confidence,
            rationale=Rationale(category=RationaleCategory.UI_TASK, reasoning="test"),
        )

    def test_executable_never_blocks(self) -> None:
        """
        EXECUTABLE label must pass through regardless of confidence.
        """

        verdict = self._verdict(label=QualificationLabel.EXECUTABLE, confidence=1.0)
        self.assertFalse(verdict.should_block(floor=self.__configuration.confidence))

    def test_probably_executable_never_blocks(self) -> None:
        """
        PROBABLY_EXECUTABLE never blocks even at very high confidence.
        """

        verdict = self._verdict(label=QualificationLabel.PROBABLY_EXECUTABLE, confidence=0.99)
        self.assertFalse(verdict.should_block(floor=self.__configuration.confidence))

    def test_probably_not_executable_never_blocks(self) -> None:
        """
        PROBABLY_NOT_EXECUTABLE never blocks; only NOT_EXECUTABLE can cross the gate.
        """

        verdict = self._verdict(label=QualificationLabel.PROBABLY_NOT_EXECUTABLE, confidence=0.99)
        self.assertFalse(verdict.should_block(floor=self.__configuration.confidence))

    def test_not_executable_below_floor_does_not_block(self) -> None:
        """
        NOT_EXECUTABLE with confidence under the floor passes through (bias toward allow).
        """

        verdict = self._verdict(
            label=QualificationLabel.NOT_EXECUTABLE,
            confidence=self.__configuration.confidence - 0.01,
        )
        self.assertFalse(verdict.should_block(floor=self.__configuration.confidence))

    def test_not_executable_at_floor_blocks(self) -> None:
        """
        NOT_EXECUTABLE exactly at the floor must block (inclusive boundary).
        """

        verdict = self._verdict(
            label=QualificationLabel.NOT_EXECUTABLE,
            confidence=self.__configuration.confidence,
        )
        self.assertTrue(verdict.should_block(floor=self.__configuration.confidence))

    def test_not_executable_high_confidence_blocks(self) -> None:
        """
        NOT_EXECUTABLE well above the floor must block.
        """

        verdict = self._verdict(label=QualificationLabel.NOT_EXECUTABLE, confidence=0.99)
        self.assertTrue(verdict.should_block(floor=self.__configuration.confidence))

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

    def test_floor_override_changes_decision(self) -> None:
        """
        Lower confidence floor must block a verdict that would pass at the default floor.
        """

        verdict = self._verdict(
            label=QualificationLabel.NOT_EXECUTABLE,
            confidence=self.__configuration.confidence - 0.05,
        )
        self.assertFalse(verdict.should_block(floor=self.__configuration.confidence))
        self.assertTrue(verdict.should_block(floor=self.__configuration.confidence - 0.10))


if __name__ == "__main__":
    unittest.main()
