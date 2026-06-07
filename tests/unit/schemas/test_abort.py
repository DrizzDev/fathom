from __future__ import annotations

import unittest

from pydantic import ValidationError

from fathom.constants.abort import (
    DEFAULT_ABORT_CONFIDENCE_FLOOR,
    DEFAULT_ABORT_DETECTOR_MODEL,
    DEFAULT_ABORT_FALLBACK_SIMILARITY_FLOOR,
)
from fathom.schemas.abort import (
    AbortConfidenceConfiguration,
    AbortDecision,
    AbortDetectorConfiguration,
    AbortDetectorResponse,
    AbortFallbackConfiguration,
    AbortInferenceConfiguration,
)


class AbortDecisionTest(unittest.TestCase):
    """
    Pins :class:`AbortDecision` field validation and immutability.
    """

    def test_default_fallback_is_false(self) -> None:
        """
        ``fallback`` defaults to False so consumers can rely on absence == primary verdict.
        """

        decision = AbortDecision(aborted=False, confidence=0.5)

        self.assertFalse(decision.fallback)

    def test_confidence_above_one_is_rejected(self) -> None:
        """
        Out-of-range confidence raises :class:`ValidationError`.
        """

        with self.assertRaises(ValidationError):
            AbortDecision(aborted=True, confidence=1.5)

    def test_decision_is_frozen(self) -> None:
        """
        Frozen Pydantic model rejects in-place mutation.
        """

        decision = AbortDecision(aborted=True, confidence=0.9)

        with self.assertRaises(ValidationError):
            decision.aborted = False  # type: ignore[misc]

    def test_extra_field_is_rejected(self) -> None:
        """
        Unknown fields are rejected so silent schema drift cannot slip past.
        """

        with self.assertRaises(ValidationError):
            AbortDecision(aborted=True, confidence=0.9, unexpected="x")  # type: ignore[call-arg]


class AbortDetectorResponseTest(unittest.TestCase):
    """
    Pins :class:`AbortDetectorResponse` boundary parsing.
    """

    def test_valid_payload_parses(self) -> None:
        """
        Canonical LLM response shape parses cleanly.
        """

        parsed = AbortDetectorResponse.model_validate({"aborted": True, "confidence": 0.92})

        self.assertTrue(parsed.aborted)
        self.assertAlmostEqual(parsed.confidence, 0.92)

    def test_missing_required_field_raises(self) -> None:
        """
        Missing required field triggers a validation error.
        """

        with self.assertRaises(ValidationError):
            AbortDetectorResponse.model_validate({"aborted": True})

    def test_confidence_below_zero_is_rejected(self) -> None:
        """
        Negative confidence is rejected at the boundary.
        """

        with self.assertRaises(ValidationError):
            AbortDetectorResponse.model_validate({"aborted": False, "confidence": -0.1})


class AbortConfigurationDefaultsTest(unittest.TestCase):
    """
    Pins default values exposed by abort-related configuration models.
    """

    def test_inference_defaults_match_constants(self) -> None:
        """
        Inference defaults sourced from constants/abort to prevent silent drift.
        """

        config = AbortInferenceConfiguration()

        self.assertEqual(config.model, DEFAULT_ABORT_DETECTOR_MODEL)

    def test_confidence_floor_default_matches_constant(self) -> None:
        """
        Confidence floor default mirrors the named constant.
        """

        config = AbortConfidenceConfiguration()

        self.assertAlmostEqual(config.floor, DEFAULT_ABORT_CONFIDENCE_FLOOR)

    def test_fallback_similarity_floor_default_matches_constant(self) -> None:
        """
        Fallback similarity floor default mirrors the named constant.
        """

        config = AbortFallbackConfiguration()

        self.assertAlmostEqual(config.similarity_floor, DEFAULT_ABORT_FALLBACK_SIMILARITY_FLOOR)

    def test_detector_configuration_nests_children(self) -> None:
        """
        Top-level configuration composes the confidence / fallback / inference children.
        """

        config = AbortDetectorConfiguration()

        self.assertIsInstance(config.fallback, AbortFallbackConfiguration)
        self.assertIsInstance(config.inference, AbortInferenceConfiguration)
        self.assertIsInstance(config.confidence, AbortConfidenceConfiguration)
