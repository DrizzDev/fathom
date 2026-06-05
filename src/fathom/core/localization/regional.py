from __future__ import annotations

from difflib import SequenceMatcher
from typing import Optional, Tuple

from fathom.constants.localization import (
    LAYOUT_MIN_WORD_LENGTH_FOR_FUZZ,
    LAYOUT_PER_WORD_SIMILARITY_THRESHOLD,
    RegionalEvidenceDecision,
)
from fathom.core.localization.matcher import OcrPhraseMatcher
from fathom.schemas.actions import Bounds
from fathom.schemas.localization import (
    RegionalEvidenceConfiguration,
    RegionalEvidenceMetrics,
    RegionalEvidenceProposal,
    RegionalEvidenceVerdict,
)
from fathom.schemas.observation import ElementSource, PerceivedElement


class RegionalEvidenceMatcher:
    """
    Score planner-emitted bounds against perceived OCR + geometry evidence
    inside the region. Domain-pure: no I/O, no logging, deterministic.
    """

    def __init__(
        self,
        *,
        phrase_matcher: OcrPhraseMatcher,
        configuration: Optional[RegionalEvidenceConfiguration] = None,
    ) -> None:
        """
        Bind the matcher to its phrase clusterer and threshold configuration.
        """

        self.__phrase_matcher = phrase_matcher
        self.__configuration = (
            configuration if configuration is not None else RegionalEvidenceConfiguration()
        )

    @property
    def configuration(self) -> RegionalEvidenceConfiguration:
        """
        Expose the immutable configuration for logging by outer layers.
        """

        return self.__configuration

    def evaluate(
        self,
        *,
        target: str,
        bounds: Bounds,
        elements: Tuple[PerceivedElement, ...],
    ) -> RegionalEvidenceVerdict:
        """
        Score planner bounds against in-region OCR evidence, returning a
        verdict carrying the decision, the math, and the proposal when resolved.
        """

        in_region_token_count = sum(
            1 for element in elements if self.__centroid_inside(element=element, bounds=bounds)
        )

        target_words = self.__words(value=target)
        if not target_words:
            return RegionalEvidenceVerdict(
                phrase=None,
                proposal=None,
                cluster_token_count=0,
                metrics=RegionalEvidenceMetrics.zero(),
                in_region_token_count=in_region_token_count,
                decision=RegionalEvidenceDecision.EMPTY_TARGET,
            )

        cluster = self.__phrase_matcher.find_best_match_within(
            target=target,
            bounds=bounds,
            elements=elements,
        )
        if cluster is None:
            return RegionalEvidenceVerdict(
                phrase=None,
                proposal=None,
                cluster_token_count=0,
                metrics=RegionalEvidenceMetrics.zero(),
                in_region_token_count=in_region_token_count,
                decision=RegionalEvidenceDecision.NO_IN_REGION_CLUSTER,
            )

        recall, density = self.__recall_and_density(
            phrase=cluster.text,
            target_words=target_words,
        )
        iou = self.__best_iou(bounds=bounds, elements=elements)
        containment = self.__best_containment(bounds=bounds, elements=elements)

        fused = self.__fused_score(
            iou=iou,
            recall=recall,
            density=density,
            containment=containment,
        )
        metrics = RegionalEvidenceMetrics(
            iou=iou,
            fused=fused,
            recall=recall,
            density=density,
            containment=containment,
        )

        if recall < self.__configuration.recall:
            return RegionalEvidenceVerdict(
                proposal=None,
                metrics=metrics,
                phrase=cluster.text,
                cluster_token_count=cluster.token_count,
                in_region_token_count=in_region_token_count,
                decision=RegionalEvidenceDecision.RECALL_BELOW_FLOOR,
            )

        if density < self.__configuration.density:
            return RegionalEvidenceVerdict(
                proposal=None,
                metrics=metrics,
                phrase=cluster.text,
                cluster_token_count=cluster.token_count,
                in_region_token_count=in_region_token_count,
                decision=RegionalEvidenceDecision.DENSITY_BELOW_FLOOR,
            )

        geometric_passes = (
            containment >= self.__configuration.containment or iou >= self.__configuration.iou
        )
        if not geometric_passes:
            return RegionalEvidenceVerdict(
                proposal=None,
                metrics=metrics,
                phrase=cluster.text,
                cluster_token_count=cluster.token_count,
                in_region_token_count=in_region_token_count,
                decision=RegionalEvidenceDecision.NO_GEOMETRIC_SIGNAL,
            )

        if fused < self.__configuration.floor:
            return RegionalEvidenceVerdict(
                proposal=None,
                metrics=metrics,
                phrase=cluster.text,
                cluster_token_count=cluster.token_count,
                in_region_token_count=in_region_token_count,
                decision=RegionalEvidenceDecision.FUSED_SCORE_BELOW_FLOOR,
            )

        proposal = RegionalEvidenceProposal(
            iou=iou,
            score=fused,
            recall=recall,
            density=density,
            phrase=cluster.text,
            bounds=cluster.bounds,
            containment=containment,
        )
        return RegionalEvidenceVerdict(
            metrics=metrics,
            proposal=proposal,
            phrase=cluster.text,
            cluster_token_count=cluster.token_count,
            decision=RegionalEvidenceDecision.RESOLVED,
            in_region_token_count=in_region_token_count,
        )

    @staticmethod
    def __centroid_inside(*, element: PerceivedElement, bounds: Bounds) -> bool:
        """
        Return whether the element centroid lies inside ``bounds``.
        """

        centroid_x = element.bounds.x + element.bounds.width / 2
        centroid_y = element.bounds.y + element.bounds.height / 2

        return (
            bounds.x <= centroid_x <= bounds.x + bounds.width
            and bounds.y <= centroid_y <= bounds.y + bounds.height
        )

    def __recall_and_density(
        self,
        *,
        phrase: str,
        target_words: Tuple[str, ...],
    ) -> Tuple[float, float]:
        """
        Return ``(target_recall, phrase_density)`` over normalized word sets.
        """

        phrase_words = self.__words(value=phrase)

        if not phrase_words:
            return 0.0, 0.0

        matched_target = sum(
            1
            for word in target_words
            if any(self.__equal(first=word, second=other) for other in phrase_words)
        )

        matched_phrase = sum(
            1
            for word in phrase_words
            if any(self.__equal(first=word, second=other) for other in target_words)
        )

        recall = matched_target / len(target_words)
        density = matched_phrase / len(phrase_words)

        return recall, density

    def __equal(self, *, first: str, second: str) -> bool:
        """
        Per-word equality with fuzzy fallback for tokens above the length floor.
        """

        if first == second:
            return True

        if (
            len(first) < LAYOUT_MIN_WORD_LENGTH_FOR_FUZZ
            or len(second) < LAYOUT_MIN_WORD_LENGTH_FOR_FUZZ
        ):
            return False

        ratio = SequenceMatcher(None, first, second).ratio()
        return ratio >= LAYOUT_PER_WORD_SIMILARITY_THRESHOLD

    def __best_containment(
        self,
        *,
        bounds: Bounds,
        elements: Tuple[PerceivedElement, ...],
    ) -> float:
        """
        Return the maximum fraction of any single in-region element's area
        that lies inside the planner-emitted bounds.
        """

        best = 0.0

        for element in elements:
            if element.source != ElementSource.OCR:
                continue

            area = element.bounds.width * element.bounds.height
            if area <= 0:
                continue

            intersection = self.__intersection_area(left=bounds, right=element.bounds)
            ratio = intersection / area

            if ratio > best:
                best = ratio

        return best

    def __best_iou(
        self,
        *,
        bounds: Bounds,
        elements: Tuple[PerceivedElement, ...],
    ) -> float:
        """
        Return the maximum IoU between ``bounds`` and any in-region element.
        """

        best = 0.0

        for element in elements:
            if element.source != ElementSource.OCR:
                continue

            score = self.__iou(left=bounds, right=element.bounds)
            if score > best:
                best = score

        return best

    @staticmethod
    def __intersection_area(*, left: Bounds, right: Bounds) -> float:
        """
        Return the pixel-area of the rectangle intersection (zero when disjoint).
        """

        x0 = max(left.x, right.x)
        y0 = max(left.y, right.y)
        x1 = min(left.x + left.width, right.x + right.width)
        y1 = min(left.y + left.height, right.y + right.height)

        if x1 <= x0 or y1 <= y0:
            return 0.0

        return float((x1 - x0) * (y1 - y0))

    @classmethod
    def __iou(cls, *, left: Bounds, right: Bounds) -> float:
        """
        Return intersection-over-union for two axis-aligned bounding rectangles.
        """

        intersection = cls.__intersection_area(left=left, right=right)

        if intersection <= 0.0:
            return 0.0

        left_area = left.width * left.height
        right_area = right.width * right.height
        union = left_area + right_area - intersection

        if union <= 0:
            return 0.0

        return intersection / union

    def __fused_score(
        self,
        *,
        iou: float,
        recall: float,
        density: float,
        containment: float,
    ) -> float:
        """
        Weighted combination of semantic and geometric signals on the unit interval.
        """

        weights = self.__configuration.weights
        return (
            weights.recall * recall
            + weights.density * density
            + weights.containment * containment
            + weights.iou * iou
        )

    @staticmethod
    def __words(*, value: str) -> Tuple[str, ...]:
        """
        Normalize a string into a tuple of lower-cased word tokens.
        """

        if not value:
            return ()

        return tuple(value.strip().lower().split())
