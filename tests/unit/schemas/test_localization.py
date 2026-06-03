from __future__ import annotations

import unittest

from pydantic import ValidationError

from fathom.constants.localization import LocalizationGridScale
from fathom.schemas.localization import VisionLocalizationPayload


class VisionLocalizationPayloadTest(unittest.TestCase):
    """
    Pins the constrained-decoding contract the vision-localizer adapter relies on.
    """

    @staticmethod
    def __valid(**overrides: object) -> dict[str, object]:
        """
        Build a baseline payload mapping callers can override per assertion.
        """

        baseline: dict[str, object] = {
            "x1": 100,
            "y1": 200,
            "x2": 400,
            "y2": 500,
            "confidence": 0.9,
            "rationale": "Tight match on the visible button glyphs.",
        }
        baseline.update(overrides)
        return baseline

    def test_valid_payload_constructs(self) -> None:
        """
        A payload inside the grid bounds with positive area constructs cleanly.
        """

        payload = VisionLocalizationPayload(**self.__valid())

        self.assertEqual(payload.x1, 100)
        self.assertEqual(payload.y2, 500)
        self.assertEqual(payload.confidence, 0.9)
        self.assertFalse(payload.refused)

    def test_edge_above_grid_maximum_is_rejected(self) -> None:
        """
        Edges past the grid maximum violate the published contract.
        """

        with self.assertRaises(ValidationError):
            VisionLocalizationPayload(**self.__valid(x2=LocalizationGridScale.MAXIMUM + 1))

    def test_inverted_axis_payload_is_rejected(self) -> None:
        """
        Right edge at or before the left edge produces no usable rectangle.
        """

        with self.assertRaises(ValidationError):
            VisionLocalizationPayload(**self.__valid(x1=400, x2=400))

    def test_zero_area_non_refusal_payload_is_rejected(self) -> None:
        """
        A degenerate rectangle that is not the refusal sentinel must fail validation.
        """

        with self.assertRaises(ValidationError):
            VisionLocalizationPayload(**self.__valid(x1=500, x2=500, y1=600, y2=600))

    def test_all_zero_payload_is_refusal(self) -> None:
        """
        The canonical refusal sentinel constructs cleanly and reports refused.
        """

        refusal = VisionLocalizationPayload(
            x1=0,
            y1=0,
            x2=0,
            y2=0,
            confidence=0.0,
            rationale="Target not visible.",
        )

        self.assertTrue(refusal.refused)

    def test_confidence_above_unit_interval_is_rejected(self) -> None:
        """
        Confidence outside the closed unit interval must not reach the ensemble.
        """

        with self.assertRaises(ValidationError):
            VisionLocalizationPayload(**self.__valid(confidence=1.5))
