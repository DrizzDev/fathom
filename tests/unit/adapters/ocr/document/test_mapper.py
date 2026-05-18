from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any, Tuple

from fathom.adapters.ocr.document.mapper import DocumentAiMapper
from fathom.schemas.ocr import OcrConfidence


class DocumentAiMapperTest(unittest.TestCase):
    """
    Pins for the Document AI response → typed OcrToken translation contract.
    """

    @staticmethod
    def __segment(*, start: int, end: int) -> SimpleNamespace:
        return SimpleNamespace(start_index=start, end_index=end)

    @staticmethod
    def __vertex(*, x: float, y: float) -> SimpleNamespace:
        return SimpleNamespace(x=x, y=y)

    @classmethod
    def __layout(
        cls,
        *,
        text_segments: Tuple[SimpleNamespace, ...],
        confidence: float,
        normalized_vertices: Tuple[SimpleNamespace, ...] = (),
        vertices: Tuple[SimpleNamespace, ...] = (),
    ) -> SimpleNamespace:
        return SimpleNamespace(
            text_anchor=SimpleNamespace(text_segments=text_segments),
            confidence=confidence,
            bounding_poly=SimpleNamespace(
                normalized_vertices=normalized_vertices,
                vertices=vertices,
            ),
        )

    @staticmethod
    def __token(*, layout: SimpleNamespace) -> SimpleNamespace:
        return SimpleNamespace(layout=layout)

    @staticmethod
    def __document(*, text: str, tokens: Tuple[SimpleNamespace, ...]) -> SimpleNamespace:
        return SimpleNamespace(text=text, pages=[SimpleNamespace(tokens=tokens)])

    def test_normalized_vertices_scale_to_pixel_bounds(self) -> None:
        """
        Normalized vertices in [0, 1] must scale by the supplied width and height.
        """

        layout = self.__layout(
            text_segments=(self.__segment(start=0, end=5),),
            confidence=0.93,
            normalized_vertices=(
                self.__vertex(x=0.1, y=0.2),
                self.__vertex(x=0.5, y=0.2),
                self.__vertex(x=0.5, y=0.4),
                self.__vertex(x=0.1, y=0.4),
            ),
        )
        document = self.__document(text="Hello", tokens=(self.__token(layout=layout),))

        tokens = DocumentAiMapper().map_document(document=document, width=1000, height=2000)

        self.assertEqual(len(tokens), 1)
        token = tokens[0]
        self.assertEqual(token.text, "Hello")
        self.assertEqual((token.bounds.x, token.bounds.y), (100, 400))
        self.assertEqual((token.bounds.width, token.bounds.height), (400, 400))
        self.assertEqual(token.confidence, OcrConfidence.HIGH)

    def test_absolute_vertices_used_when_normalized_absent(self) -> None:
        """
        Pre-pixel vertices are consumed verbatim when no normalized polygon is present.
        """

        layout = self.__layout(
            text_segments=(self.__segment(start=0, end=5),),
            confidence=0.6,
            normalized_vertices=(),
            vertices=(
                self.__vertex(x=10, y=20),
                self.__vertex(x=60, y=20),
                self.__vertex(x=60, y=80),
                self.__vertex(x=10, y=80),
            ),
        )
        document = self.__document(text="Hello", tokens=(self.__token(layout=layout),))

        tokens = DocumentAiMapper().map_document(document=document, width=200, height=200)

        bounds = tokens[0].bounds
        self.assertEqual((bounds.x, bounds.y, bounds.width, bounds.height), (10, 20, 50, 60))

    def test_absolute_vertices_clamped_against_image_rectangle(self) -> None:
        """
        Preview-processor responses with out-of-image vertices must be clamped.

        Some Document AI preview processors emit absolute coordinates wider
        than the image; downstream consumers expect pixel bounds inside the
        screen rectangle.
        """

        layout = self.__layout(
            text_segments=(self.__segment(start=0, end=4),),
            confidence=0.9,
            vertices=(
                self.__vertex(x=50, y=50),
                self.__vertex(x=500, y=50),
                self.__vertex(x=500, y=500),
                self.__vertex(x=50, y=500),
            ),
        )
        document = self.__document(text="Word", tokens=(self.__token(layout=layout),))

        tokens = DocumentAiMapper().map_document(document=document, width=200, height=200)

        bounds = tokens[0].bounds
        self.assertLessEqual(bounds.x + bounds.width, 200)
        self.assertLessEqual(bounds.y + bounds.height, 200)

    def test_token_dropped_when_text_resolves_to_empty(self) -> None:
        layout = self.__layout(
            text_segments=(),
            confidence=0.9,
            normalized_vertices=(self.__vertex(x=0.0, y=0.0), self.__vertex(x=0.1, y=0.1)),
        )
        document = self.__document(text="anything", tokens=(self.__token(layout=layout),))

        self.assertEqual(
            DocumentAiMapper().map_document(document=document, width=100, height=100),
            (),
        )

    def test_token_dropped_when_bounds_collapse_to_zero_area(self) -> None:
        layout = self.__layout(
            text_segments=(self.__segment(start=0, end=4),),
            confidence=0.9,
            normalized_vertices=(
                self.__vertex(x=0.5, y=0.5),
                self.__vertex(x=0.5, y=0.5),
            ),
        )
        document = self.__document(text="Word", tokens=(self.__token(layout=layout),))

        self.assertEqual(
            DocumentAiMapper().map_document(document=document, width=100, height=100),
            (),
        )

    def test_provider_score_maps_to_coarse_confidence_band(self) -> None:
        """
        Provider confidence ≥ HIGH floor → HIGH band; ≥ MEDIUM floor → MEDIUM; else LOW.
        """

        def __build(score: float) -> Any:
            return self.__document(
                text="Word",
                tokens=(
                    self.__token(
                        layout=self.__layout(
                            text_segments=(self.__segment(start=0, end=4),),
                            confidence=score,
                            normalized_vertices=(
                                self.__vertex(x=0.0, y=0.0),
                                self.__vertex(x=0.2, y=0.0),
                                self.__vertex(x=0.2, y=0.1),
                                self.__vertex(x=0.0, y=0.1),
                            ),
                        )
                    ),
                ),
            )

        mapper = DocumentAiMapper()
        high = mapper.map_document(document=__build(0.95), width=100, height=100)[0]
        medium = mapper.map_document(document=__build(0.7), width=100, height=100)[0]
        low = mapper.map_document(document=__build(0.2), width=100, height=100)[0]

        self.assertEqual(high.confidence, OcrConfidence.HIGH)
        self.assertEqual(medium.confidence, OcrConfidence.MEDIUM)
        self.assertEqual(low.confidence, OcrConfidence.LOW)

    def test_multi_segment_text_concatenates_in_order(self) -> None:
        layout = self.__layout(
            text_segments=(
                self.__segment(start=0, end=5),
                self.__segment(start=6, end=11),
            ),
            confidence=0.9,
            normalized_vertices=(
                self.__vertex(x=0.0, y=0.0),
                self.__vertex(x=0.5, y=0.0),
                self.__vertex(x=0.5, y=0.2),
                self.__vertex(x=0.0, y=0.2),
            ),
        )
        document = self.__document(text="Hello World", tokens=(self.__token(layout=layout),))

        tokens = DocumentAiMapper().map_document(document=document, width=200, height=200)

        self.assertEqual(tokens[0].text, "HelloWorld")
