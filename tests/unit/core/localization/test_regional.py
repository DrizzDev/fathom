from __future__ import annotations

import unittest
from typing import Tuple

from fathom.constants.localization import RegionalEvidenceDecision
from fathom.core.localization.matcher import OcrPhraseMatcher
from fathom.core.localization.regional import RegionalEvidenceMatcher
from fathom.schemas.actions import Bounds, CoordinateSource, CoordinateSystem
from fathom.schemas.localization import RegionalEvidenceConfiguration
from fathom.schemas.observation import ElementRole, ElementSource, PerceivedElement


class RegionalEvidenceMatcherTest(unittest.TestCase):
    """
    Pins the triple-gated decision (spatial pre-filter, row-cluster merge,
    recall + density + containment / IoU) of the regional evidence matcher.
    """

    @staticmethod
    def __ocr_element(
        *,
        text: str,
        x: int,
        y: int,
        width: int,
        height: int,
        identifier: str,
        confidence: float = 0.95,
    ) -> PerceivedElement:
        """
        Build one OCR-sourced perceived element with stable defaults.
        """

        return PerceivedElement(
            text=text,
            parent=None,
            tappable=False,
            label_id=identifier,
            confidence=confidence,
            identifier=identifier,
            role=ElementRole.TEXT,
            source=ElementSource.OCR,
            bounds=Bounds(
                x=x,
                y=y,
                width=width,
                height=height,
                source=CoordinateSource.OCR,
                coordinate_system=CoordinateSystem.DEVICE_PIXEL,
            ),
        )

    @staticmethod
    def __fragmented_phrase_row() -> Tuple[PerceivedElement, ...]:
        """
        Five OCR tokens on a single baseline forming one merged phrase.
        """

        ocr = RegionalEvidenceMatcherTest.__ocr_element

        return (
            ocr(text="Buy", x=327, y=2034, width=70, height=33, identifier="ocr_1"),
            ocr(text="tickets", x=405, y=2033, width=124, height=33, identifier="ocr_2"),
            ocr(text="from", x=540, y=2033, width=84, height=33, identifier="ocr_3"),
            ocr(text="$", x=637, y=2033, width=23, height=31, identifier="ocr_4"),
            ocr(text="0.00", x=664, y=2033, width=84, height=31, identifier="ocr_5"),
        )

    @staticmethod
    def __enclosing_model_bounds() -> Bounds:
        """
        Model bounds enclosing the fragmented OCR row.
        """

        return Bounds(
            x=31,
            y=1989,
            width=937,
            height=73,
            source=CoordinateSource.MODEL,
            coordinate_system=CoordinateSystem.DEVICE_PIXEL,
        )

    def __matcher(self) -> RegionalEvidenceMatcher:
        """
        Build a matcher with default configuration sourced from constants.
        """

        return RegionalEvidenceMatcher(phrase_matcher=OcrPhraseMatcher())

    def test_target_prefix_resolves_to_phrase_cluster(self) -> None:
        """
        A target prefix of the merged phrase resolves to the cluster's tight bounds.
        """

        verdict = self.__matcher().evaluate(
            target="Buy tickets",
            bounds=self.__enclosing_model_bounds(),
            elements=self.__fragmented_phrase_row(),
        )

        self.assertTrue(verdict.resolved)
        self.assertIsNotNone(verdict.proposal)
        self.assertEqual(verdict.proposal.bounds.x, 327)
        self.assertEqual(verdict.proposal.bounds.y, 2033)
        self.assertEqual(verdict.cluster_token_count, 5)
        self.assertEqual(verdict.in_region_token_count, 5)
        self.assertAlmostEqual(verdict.metrics.recall, 1.0)
        self.assertEqual(verdict.proposal.bounds.width, 421)
        self.assertEqual(verdict.proposal.bounds.height, 34)
        self.assertEqual(verdict.phrase, "Buy tickets from $ 0.00")
        self.assertEqual(verdict.decision, RegionalEvidenceDecision.RESOLVED)

    def test_full_phrase_target_passes_density_floor(self) -> None:
        """
        A target carrying every phrase word lifts density above the floor.
        """

        verdict = self.__matcher().evaluate(
            target="Buy tickets from $0.00",
            bounds=self.__enclosing_model_bounds(),
            elements=self.__fragmented_phrase_row(),
        )

        self.assertGreaterEqual(verdict.metrics.fused, 0.7)
        self.assertGreaterEqual(verdict.metrics.density, 0.7)
        self.assertEqual(verdict.decision, RegionalEvidenceDecision.RESOLVED)

    def test_token_outside_region_dropped_by_spatial_filter(self) -> None:
        """
        Tokens whose centroid lies outside the model bounds never enter scoring.
        """

        elsewhere = self.__ocr_element(
            x=10,
            y=10,
            width=200,
            height=40,
            text="Buy tickets",
            identifier="ocr_outside",
        )

        verdict = self.__matcher().evaluate(
            target="Buy tickets",
            elements=(elsewhere,),
            bounds=self.__enclosing_model_bounds(),
        )

        self.assertIsNone(verdict.proposal)
        self.assertEqual(verdict.in_region_token_count, 0)
        self.assertEqual(verdict.decision, RegionalEvidenceDecision.NO_IN_REGION_CLUSTER)

    def test_recall_below_floor_abstains(self) -> None:
        """
        Insufficient target-word recall inside the region abstains.
        """

        partial = self.__ocr_element(
            text="Buy", x=327, y=2034, width=70, height=33, identifier="ocr_partial"
        )

        verdict = self.__matcher().evaluate(
            target="Buy tickets",
            elements=(partial,),
            bounds=self.__enclosing_model_bounds(),
        )

        self.assertIsNone(verdict.proposal)
        self.assertAlmostEqual(verdict.metrics.recall, 0.5)
        self.assertEqual(verdict.decision, RegionalEvidenceDecision.RECALL_BELOW_FLOOR)

    def test_density_floor_rejects_dilute_phrase(self) -> None:
        """
        A cluster whose density of target words is below the floor is rejected.
        """

        configuration = RegionalEvidenceConfiguration(
            iou=0.0,
            floor=0.5,
            recall=0.8,
            density=0.9,
            containment=0.0,
        )
        matcher = RegionalEvidenceMatcher(
            phrase_matcher=OcrPhraseMatcher(), configuration=configuration
        )

        verdict = matcher.evaluate(
            target="Buy tickets",
            bounds=self.__enclosing_model_bounds(),
            elements=self.__fragmented_phrase_row(),
        )

        self.assertIsNone(verdict.proposal)
        self.assertLess(verdict.metrics.density, 0.9)
        self.assertEqual(verdict.decision, RegionalEvidenceDecision.DENSITY_BELOW_FLOOR)

    def test_missing_geometric_signal_rejects_proposal(self) -> None:
        """
        Failing both containment and IoU floors rejects even when semantics pass.
        """

        configuration = RegionalEvidenceConfiguration(
            iou=0.99,
            floor=0.0,
            recall=0.8,
            density=0.3,
            containment=0.99,
        )
        matcher = RegionalEvidenceMatcher(
            phrase_matcher=OcrPhraseMatcher(), configuration=configuration
        )

        verdict = matcher.evaluate(
            target="Buy tickets",
            bounds=self.__enclosing_model_bounds(),
            elements=self.__fragmented_phrase_row(),
        )

        self.assertIsNone(verdict.proposal)
        self.assertEqual(verdict.decision, RegionalEvidenceDecision.NO_GEOMETRIC_SIGNAL)

    def test_empty_target_short_circuits_with_decision(self) -> None:
        """
        A blank target short-circuits and surfaces ``EMPTY_TARGET`` for log attribution.
        """

        verdict = self.__matcher().evaluate(
            target="   ",
            bounds=self.__enclosing_model_bounds(),
            elements=self.__fragmented_phrase_row(),
        )

        self.assertIsNone(verdict.proposal)
        self.assertEqual(verdict.in_region_token_count, 5)
        self.assertEqual(verdict.decision, RegionalEvidenceDecision.EMPTY_TARGET)

    def test_resolved_verdict_carries_all_observability_metrics(self) -> None:
        """
        Every metric and cluster field is populated for downstream structured logging.
        """

        verdict = self.__matcher().evaluate(
            target="Buy tickets",
            bounds=self.__enclosing_model_bounds(),
            elements=self.__fragmented_phrase_row(),
        )

        self.assertGreater(verdict.metrics.iou, 0.0)
        self.assertGreater(verdict.metrics.fused, 0.0)
        self.assertGreater(verdict.metrics.recall, 0.0)
        self.assertGreater(verdict.metrics.density, 0.0)
        self.assertEqual(verdict.cluster_token_count, 5)
        self.assertEqual(verdict.in_region_token_count, 5)
        self.assertGreater(verdict.metrics.containment, 0.0)
        self.assertEqual(verdict.phrase, "Buy tickets from $ 0.00")
        self.assertEqual(verdict.decision, RegionalEvidenceDecision.RESOLVED)


if __name__ == "__main__":
    unittest.main()
