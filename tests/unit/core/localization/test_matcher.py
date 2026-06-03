from __future__ import annotations

import unittest
from typing import Tuple

from fathom.core.localization.matcher import OcrPhraseMatcher
from fathom.schemas.actions import Bounds, CoordinateSource, CoordinateSystem
from fathom.schemas.localization import LayoutMatchConfiguration
from fathom.schemas.observation import ElementRole, ElementSource, PerceivedElement


class OcrPhraseMatcherTest(unittest.TestCase):
    """
    Pins clustering, scoring, and rejection invariants for the OCR phrase matcher.
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
            parent=None,
            tappable=False,
            text=text,
            label_id=identifier,
            bounds=Bounds(
                x=x,
                y=y,
                width=width,
                height=height,
                source=CoordinateSource.OCR,
                coordinate_system=CoordinateSystem.DEVICE_PIXEL,
            ),
            role=ElementRole.TEXT,
            source=ElementSource.OCR,
            identifier=identifier,
            confidence=confidence,
        )

    @staticmethod
    def __earn_spoons_row() -> Tuple[PerceivedElement, ...]:
        """
        Reproduce the Free + Offers tokens from the Earn Spoons screen of run 38890d03.
        """

        return (
            OcrPhraseMatcherTest.__ocr_element(
                text="Free",
                x=1040,
                y=705,
                width=115,
                height=41,
                identifier="ocr_27",
            ),
            OcrPhraseMatcherTest.__ocr_element(
                text="Offers",
                x=1224,
                y=703,
                width=171,
                height=45,
                identifier="ocr_28",
            ),
        )

    def test_phrase_match_clusters_free_and_offers_into_one_phrase(self) -> None:
        """
        Production failure case: two adjacent OCR words form the Free Offers phrase.
        """

        matcher = OcrPhraseMatcher()

        match = matcher.find_best_match(
            target="Free Offers",
            elements=self.__earn_spoons_row(),
        )

        self.assertIsNotNone(match)
        assert match is not None
        self.assertEqual(match.text, "Free Offers")
        self.assertEqual(match.token_count, 2)
        self.assertEqual(match.bounds.x, 1040)
        self.assertEqual(match.bounds.width, 1395 - 1040)
        self.assertGreaterEqual(match.score, 0.95)

    def test_single_token_target_still_matches_when_phrase_is_single_word(self) -> None:
        """
        A one-word target must keep matching a one-word OCR phrase exactly as before.
        """

        matcher = OcrPhraseMatcher()
        elements = (
            self.__ocr_element(
                text="Submit", x=100, y=200, width=140, height=44, identifier="ocr_1"
            ),
        )

        match = matcher.find_best_match(target="Submit", elements=elements)

        self.assertIsNotNone(match)
        assert match is not None
        self.assertEqual(match.text, "Submit")
        self.assertEqual(match.score, 1.0)

    def test_cross_row_tokens_never_merge(self) -> None:
        """
        Tokens on different rows must not be joined into one phrase even if their texts spell the target.
        """

        matcher = OcrPhraseMatcher()
        elements = (
            self.__ocr_element(
                text="Free", x=1040, y=100, width=115, height=41, identifier="ocr_1"
            ),
            self.__ocr_element(
                text="Offers", x=1224, y=800, width=171, height=45, identifier="ocr_2"
            ),
        )

        match = matcher.find_best_match(target="Free Offers", elements=elements)

        self.assertIsNone(match)

    def test_large_horizontal_gap_breaks_cluster(self) -> None:
        """
        Tokens whose gap exceeds the configured ratio of token height must not merge.
        """

        matcher = OcrPhraseMatcher()
        elements = (
            self.__ocr_element(text="Free", x=10, y=500, width=115, height=41, identifier="ocr_1"),
            self.__ocr_element(
                text="Offers", x=2000, y=500, width=171, height=45, identifier="ocr_2"
            ),
        )

        match = matcher.find_best_match(target="Free Offers", elements=elements)

        self.assertIsNone(match)

    def test_low_precision_phrase_rejected_below_threshold(self) -> None:
        """
        Target embedded in a much longer phrase fails the precision component of F1.
        """

        matcher = OcrPhraseMatcher()
        elements = (
            self.__ocr_element(text="Tap", x=100, y=200, width=80, height=36, identifier="ocr_1"),
            self.__ocr_element(
                text="Continue", x=200, y=200, width=200, height=36, identifier="ocr_2"
            ),
            self.__ocr_element(text="to", x=420, y=200, width=60, height=36, identifier="ocr_3"),
            self.__ocr_element(
                text="proceed", x=500, y=200, width=170, height=36, identifier="ocr_4"
            ),
        )

        match = matcher.find_best_match(target="Continue", elements=elements)

        self.assertIsNone(match)

    def test_disjoint_target_words_yield_no_match(self) -> None:
        """
        A phrase that shares no words with the target produces no match.
        """

        matcher = OcrPhraseMatcher()
        elements = (
            self.__ocr_element(text="Save", x=100, y=200, width=120, height=40, identifier="ocr_1"),
        )

        match = matcher.find_best_match(target="Cancel", elements=elements)

        self.assertIsNone(match)

    def test_low_confidence_token_excluded_from_match(self) -> None:
        """
        Tokens below the confidence floor must not contribute to a match.
        """

        matcher = OcrPhraseMatcher()
        elements = (
            self.__ocr_element(
                text="Free",
                x=1040,
                y=705,
                width=115,
                height=41,
                identifier="ocr_27",
                confidence=0.3,
            ),
            self.__ocr_element(
                text="Offers",
                x=1224,
                y=703,
                width=171,
                height=45,
                identifier="ocr_28",
                confidence=0.3,
            ),
        )

        match = matcher.find_best_match(target="Free Offers", elements=elements)

        self.assertIsNone(match)

    def test_ocr_misread_within_similarity_floor_still_matches(self) -> None:
        """
        Single-character OCR misreads should be tolerated by the SequenceMatcher equality.
        """

        matcher = OcrPhraseMatcher()
        elements = (
            self.__ocr_element(
                text="Free", x=1040, y=705, width=115, height=41, identifier="ocr_1"
            ),
            self.__ocr_element(
                text="0ffers", x=1224, y=703, width=171, height=45, identifier="ocr_2"
            ),
        )

        match = matcher.find_best_match(target="Free Offers", elements=elements)

        self.assertIsNotNone(match)
        assert match is not None
        self.assertEqual(match.text, "Free 0ffers")

    def test_short_target_requires_exact_word_match(self) -> None:
        """
        Targets shorter than the fuzz-floor must not match similar-shaped words.
        """

        matcher = OcrPhraseMatcher()
        elements = (
            self.__ocr_element(text="Tab", x=100, y=200, width=60, height=30, identifier="ocr_1"),
        )

        match = matcher.find_best_match(target="Tap", elements=elements)

        self.assertIsNone(match)

    def test_non_ocr_elements_are_ignored(self) -> None:
        """
        The matcher only considers OCR-sourced elements regardless of their text.
        """

        matcher = OcrPhraseMatcher()
        xml_element = PerceivedElement(
            parent=None,
            tappable=True,
            text="Free Offers",
            label_id="xml_1",
            bounds=Bounds(
                x=1040,
                y=703,
                width=355,
                height=45,
                source=CoordinateSource.XML,
                coordinate_system=CoordinateSystem.DEVICE_PIXEL,
            ),
            role=ElementRole.BUTTON,
            source=ElementSource.XML,
            identifier="xml_1",
            confidence=1.0,
        )

        match = matcher.find_best_match(target="Free Offers", elements=(xml_element,))

        self.assertIsNone(match)

    def test_threshold_lowered_via_configuration_allows_softer_match(self) -> None:
        """
        Configuration injection lets callers loosen the phrase-match threshold without code changes.
        """

        matcher = OcrPhraseMatcher(
            configuration=LayoutMatchConfiguration(phrase_match_threshold=0.5),
        )
        elements = (
            self.__ocr_element(
                text="Free", x=1040, y=705, width=115, height=41, identifier="ocr_1"
            ),
            self.__ocr_element(
                text="Surveys", x=1224, y=703, width=171, height=45, identifier="ocr_2"
            ),
        )

        match = matcher.find_best_match(target="Free Offers", elements=elements)

        self.assertIsNotNone(match)
        assert match is not None
        self.assertLess(match.score, 0.8)

    def test_empty_target_returns_no_match(self) -> None:
        """
        A blank target text must short-circuit without scanning OCR elements.
        """

        matcher = OcrPhraseMatcher()
        elements = self.__earn_spoons_row()

        match = matcher.find_best_match(target="   ", elements=elements)

        self.assertIsNone(match)

    def test_empty_elements_returns_no_match(self) -> None:
        """
        With no OCR elements the matcher returns None instead of fabricating a match.
        """

        matcher = OcrPhraseMatcher()

        match = matcher.find_best_match(target="Free Offers", elements=())

        self.assertIsNone(match)
