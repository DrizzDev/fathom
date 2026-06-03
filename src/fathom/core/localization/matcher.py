from __future__ import annotations

from difflib import SequenceMatcher
from typing import List, Optional, Tuple

from fathom.schemas.actions import Bounds
from fathom.schemas.localization import LayoutMatchConfiguration, PhraseMatch
from fathom.schemas.observation import ElementSource, PerceivedElement


class OcrPhraseMatcher:
    """
    Clusters adjacent OCR tokens into phrases and scores them against a target.
    """

    def __init__(self, *, configuration: Optional[LayoutMatchConfiguration] = None) -> None:
        """
        Bind the matcher to its tunable configuration.
        """

        self.__configuration = (
            configuration if configuration is not None else LayoutMatchConfiguration()
        )

    @property
    def configuration(self) -> LayoutMatchConfiguration:
        """
        Expose the immutable configuration so callers can log tunable's alongside matches.
        """

        return self.__configuration

    def find_best_match(
        self,
        *,
        target: str,
        elements: Tuple[PerceivedElement, ...],
    ) -> Optional[PhraseMatch]:
        """
        Return the highest-scoring phrase match above the configured threshold.
        """

        if not (target_words := self.__words(value=target)):
            return None

        if not (ocr_elements := self.__filter_eligible(elements=elements)):
            return None

        candidates: List[PhraseMatch] = []
        for row in self.__cluster_rows(elements=ocr_elements):
            candidates.extend(self.__walk_row(row=row, target_words=target_words))

        if not candidates:
            return None

        best = max(candidates, key=lambda candidate: candidate.score)
        if best.score < self.__configuration.phrase_match_threshold:
            return None

        return best

    def __filter_eligible(
        self,
        *,
        elements: Tuple[PerceivedElement, ...],
    ) -> Tuple[PerceivedElement, ...]:
        """
        Keep OCR-sourced elements that carry text and meet the confidence floor.
        """

        floor = self.__configuration.min_token_confidence

        eligible = (
            element
            for element in elements
            if element.source == ElementSource.OCR and element.text and element.confidence >= floor
        )
        return tuple(eligible)

    def __cluster_rows(
        self,
        *,
        elements: Tuple[PerceivedElement, ...],
    ) -> List[List[PerceivedElement]]:
        """
        Group elements into rows whose vertical centres fall inside the row tolerance.
        """

        rows: List[List[PerceivedElement]] = []
        tolerance_ratio = self.__configuration.max_row_offset_ratio

        for element in elements:
            placed = False
            element_centre_y = element.bounds.y + element.bounds.height / 2

            for row in rows:
                reference = row[0]
                tolerance = reference.bounds.height * tolerance_ratio
                reference_centre_y = reference.bounds.y + reference.bounds.height / 2

                if abs(element_centre_y - reference_centre_y) <= tolerance:
                    row.append(element)
                    placed = True
                    break

            if not placed:
                rows.append([element])

        return rows

    def __walk_row(
        self,
        *,
        row: List[PerceivedElement],
        target_words: Tuple[str, ...],
    ) -> List[PhraseMatch]:
        """
        Walk one row left-to-right, joining adjacent tokens and scoring each cluster.
        """

        row.sort(key=lambda element: element.bounds.x)
        height_ratio = self.__configuration.max_height_ratio
        gap_ratio = self.__configuration.max_horizontal_gap_ratio

        active: List[PerceivedElement] = []
        clusters: List[List[PerceivedElement]] = []

        for element in row:
            if not active:
                active.append(element)
                continue

            previous = active[-1]

            height_anchor = max(previous.bounds.height, element.bounds.height)
            gap = element.bounds.x - (previous.bounds.x + previous.bounds.width)

            taller = max(previous.bounds.height, element.bounds.height)
            shorter = max(1, min(previous.bounds.height, element.bounds.height))

            within_gap = gap <= height_anchor * gap_ratio
            within_height = taller / shorter <= height_ratio

            if within_gap and within_height:
                active.append(element)
            else:
                clusters.append(active)
                active = [element]

        if active:
            clusters.append(active)

        matches: List[PhraseMatch] = []
        for cluster in clusters:
            phrase_words = self.__words(value=" ".join(element.text or "" for element in cluster))

            if (score := self.__score(target_words=target_words, phrase_words=phrase_words)) <= 0.0:
                continue

            matches.append(self.__merge(cluster=cluster, score=score))

        return matches

    def __score(
        self,
        *,
        target_words: Tuple[str, ...],
        phrase_words: Tuple[str, ...],
    ) -> float:
        """
        F1 over normalized word sets with per-word fuzzy equality.
        """

        if not target_words or not phrase_words:
            return 0.0

        matched_target = sum(
            1 for word in target_words if any(self.__equal(word, other) for other in phrase_words)
        )
        matched_phrase = sum(
            1 for word in phrase_words if any(self.__equal(word, other) for other in target_words)
        )
        if matched_target == 0 or matched_phrase == 0:
            return 0.0

        recall = matched_target / len(target_words)
        precision = matched_phrase / len(phrase_words)

        if recall + precision == 0:
            return 0.0

        return 2 * recall * precision / (recall + precision)

    def __equal(self, first: str, second: str) -> bool:
        """
        Per-word equality with SequenceMatcher fuzz above the configured similarity floor.
        """

        if first == second:
            return True

        if (
            len(first) < self.__configuration.min_word_length_for_fuzz
            or len(second) < self.__configuration.min_word_length_for_fuzz
        ):
            return False

        ratio = SequenceMatcher(None, first, second).ratio()
        return ratio >= self.__configuration.per_word_similarity_threshold

    @staticmethod
    def __words(*, value: str) -> Tuple[str, ...]:
        """
        Normalize a string into a tuple of lowercased word tokens.
        """

        if not value:
            return ()

        return tuple(value.strip().lower().split())

    @staticmethod
    def __merge(
        *,
        score: float,
        cluster: List[PerceivedElement],
    ) -> PhraseMatch:
        """
        Build the union-bounds PhraseMatch covering every element in a cluster.
        """

        first = cluster[0]
        merged_x = min(element.bounds.x for element in cluster)
        merged_y = min(element.bounds.y for element in cluster)
        right_edge = max(element.bounds.x + element.bounds.width for element in cluster)
        bottom_edge = max(element.bounds.y + element.bounds.height for element in cluster)

        bounds = first.bounds.model_copy(
            update={
                "x": merged_x,
                "y": merged_y,
                "width": right_edge - merged_x,
                "height": bottom_edge - merged_y,
            },
        )

        if not isinstance(bounds, Bounds):
            raise TypeError("Merge produced non-Bounds payload")

        return PhraseMatch(
            score=score,
            bounds=bounds,
            token_count=len(cluster),
            confidence=min(element.confidence for element in cluster),
            text=" ".join(element.text or "" for element in cluster).strip(),
        )
