from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any, Tuple

from fathom.adapters.ocr.document.mapper import DocumentAiMapper
from fathom.constants.ocr import OcrConfidence, OcrLevel


class DocumentAiMapperTest(unittest.TestCase):
    """
    Pins for the Document AI response → typed OcrToken translation contract across all layout hierarchy levels.
    """

    @staticmethod
    def __segment(*, start: int, end: int) -> SimpleNamespace:
        """
        Build one Document AI text-anchor segment over ``[start, end)``.
        """

        return SimpleNamespace(start_index=start, end_index=end)

    @staticmethod
    def __vertex(*, x: float, y: float) -> SimpleNamespace:
        """
        Build one polygon vertex with the supplied coordinates.
        """

        return SimpleNamespace(x=x, y=y)

    @classmethod
    def __layout(
        cls,
        *,
        confidence: float,
        text_segments: Tuple[SimpleNamespace, ...],
        vertices: Tuple[SimpleNamespace, ...] = (),
        normalized_vertices: Tuple[SimpleNamespace, ...] = (),
    ) -> SimpleNamespace:
        """
        Build a Document AI layout carrying a text anchor and a bounding polygon.
        """

        return SimpleNamespace(
            text_anchor=SimpleNamespace(text_segments=text_segments),
            confidence=confidence,
            bounding_poly=SimpleNamespace(
                vertices=vertices,
                normalized_vertices=normalized_vertices,
            ),
        )

    @staticmethod
    def __token(*, layout: SimpleNamespace) -> SimpleNamespace:
        """
        Wrap a layout in the page-item shape Document AI emits for tokens, lines, and paragraphs.
        """

        return SimpleNamespace(layout=layout)

    @staticmethod
    def __document(
        *,
        text: str,
        tokens: Tuple[SimpleNamespace, ...],
        lines: Tuple[SimpleNamespace, ...] = (),
        paragraphs: Tuple[SimpleNamespace, ...] = (),
    ) -> SimpleNamespace:
        """
        Build a one-page Document AI document with the supplied per-level items.
        """

        return SimpleNamespace(
            text=text,
            pages=[
                SimpleNamespace(
                    tokens=tokens,
                    lines=lines,
                    paragraphs=paragraphs,
                )
            ],
        )

    @classmethod
    def __row_layout(
        cls,
        *,
        start: int,
        end: int,
        confidence: float,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
    ) -> SimpleNamespace:
        """
        Build a layout whose text anchor spans ``[start, end)`` and whose
        normalized polygon is the rectangle (x0, y0) -> (x1, y1).
        """

        return cls.__layout(
            text_segments=(cls.__segment(start=start, end=end),),
            confidence=confidence,
            normalized_vertices=(
                cls.__vertex(x=x0, y=y0),
                cls.__vertex(x=x1, y=y0),
                cls.__vertex(x=x1, y=y1),
                cls.__vertex(x=x0, y=y1),
            ),
        )

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

        Some Document AI preview processors emit absolute coordinates wider than the image;
        downstream consumers expect pixel bounds inside the screen rectangle.
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
        """
        A layout whose text anchor carries no segments resolves to empty text and must be discarded.
        """

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
        """
        A degenerate polygon that projects to a zero-area rectangle must be discarded.
        """

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

        low = mapper.map_document(document=__build(0.2), width=100, height=100)[0]
        high = mapper.map_document(document=__build(0.95), width=100, height=100)[0]
        medium = mapper.map_document(document=__build(0.7), width=100, height=100)[0]

        self.assertEqual(low.confidence, OcrConfidence.LOW)
        self.assertEqual(high.confidence, OcrConfidence.HIGH)
        self.assertEqual(medium.confidence, OcrConfidence.MEDIUM)

    def test_token_default_level_is_token(self) -> None:
        """
        A mapped element defaults to ``OcrLevel.TOKEN`` so legacy
        consumers see the same shape as before the level field landed.
        """

        layout = self.__row_layout(start=0, end=5, confidence=0.93, x0=0.1, y0=0.2, x1=0.5, y1=0.3)
        document = self.__document(text="Hello", tokens=(self.__token(layout=layout),))

        tokens = DocumentAiMapper().map_document(document=document, width=1000, height=2000)

        self.assertEqual(tokens[0].level, OcrLevel.TOKEN)

    def test_multi_token_line_emitted_at_line_level(self) -> None:
        """
        A Document AI line that contains more than one token must surface
        as an additional :class:`OcrToken` carrying ``OcrLevel.LINE``.
        """

        token_a = self.__token(
            layout=self.__row_layout(
                start=0, end=3, confidence=0.9, x0=0.10, y0=0.50, x1=0.20, y1=0.55
            )
        )
        token_b = self.__token(
            layout=self.__row_layout(
                start=4, end=11, confidence=0.9, x0=0.22, y0=0.50, x1=0.45, y1=0.55
            )
        )
        line = self.__token(
            layout=self.__row_layout(
                start=0, end=11, confidence=0.9, x0=0.10, y0=0.50, x1=0.45, y1=0.55
            )
        )
        document = self.__document(
            text="Buy tickets",
            tokens=(token_a, token_b),
            lines=(line,),
        )

        mapped = DocumentAiMapper().map_document(document=document, width=1000, height=2000)

        levels = {element.level for element in mapped}
        self.assertIn(OcrLevel.LINE, levels)

        line_element = next(element for element in mapped if element.level is OcrLevel.LINE)

        self.assertEqual(line_element.bounds.x, 100)
        self.assertEqual(line_element.bounds.y, 1000)
        self.assertEqual(line_element.bounds.width, 350)
        self.assertEqual(line_element.bounds.height, 100)
        self.assertEqual(line_element.text, "Buy tickets")

    def test_single_token_line_suppressed(self) -> None:
        """
        Lines whose character span covers at most one token duplicate the
        token bounds and must not surface as a separate line element.
        """

        token = self.__token(
            layout=self.__row_layout(
                start=0, end=4, confidence=0.9, x0=0.1, y0=0.5, x1=0.2, y1=0.55
            )
        )
        line = self.__token(
            layout=self.__row_layout(
                start=0, end=4, confidence=0.9, x0=0.1, y0=0.5, x1=0.2, y1=0.55
            )
        )
        document = self.__document(text="Skip", tokens=(token,), lines=(line,))

        mapped = DocumentAiMapper().map_document(document=document, width=1000, height=2000)

        self.assertEqual(len(mapped), 1)
        self.assertEqual(mapped[0].level, OcrLevel.TOKEN)

    def test_multi_line_paragraph_emitted_at_paragraph_level(self) -> None:
        """
        Paragraphs that span more than one line must surface as
        an additional :class:`OcrToken` carrying ``OcrLevel.PARAGRAPH``.
        """

        token_a = self.__token(
            layout=self.__row_layout(
                start=0, end=5, confidence=0.9, x0=0.1, y0=0.5, x1=0.2, y1=0.55
            )
        )
        token_b = self.__token(
            layout=self.__row_layout(
                start=6, end=11, confidence=0.9, x0=0.1, y0=0.57, x1=0.2, y1=0.62
            )
        )
        line_a = self.__token(
            layout=self.__row_layout(
                start=0, end=5, confidence=0.9, x0=0.1, y0=0.5, x1=0.2, y1=0.55
            )
        )
        line_b = self.__token(
            layout=self.__row_layout(
                start=6, end=11, confidence=0.9, x0=0.1, y0=0.57, x1=0.2, y1=0.62
            )
        )
        paragraph = self.__token(
            layout=self.__row_layout(
                start=0, end=11, confidence=0.9, x0=0.1, y0=0.5, x1=0.2, y1=0.62
            )
        )
        document = self.__document(
            text="Hello World",
            tokens=(token_a, token_b),
            lines=(line_a, line_b),
            paragraphs=(paragraph,),
        )

        mapped = DocumentAiMapper().map_document(document=document, width=1000, height=2000)

        paragraphs = [element for element in mapped if element.level is OcrLevel.PARAGRAPH]

        self.assertEqual(len(paragraphs), 1)
        self.assertEqual(paragraphs[0].text, "Hello World")
        self.assertEqual(paragraphs[0].bounds.height, 240)

    def test_single_line_paragraph_suppressed(self) -> None:
        """
        Paragraphs that span only one line duplicate the line bounds and
        must not surface as separate paragraph elements.
        """

        token_a = self.__token(
            layout=self.__row_layout(
                start=0, end=5, confidence=0.9, x0=0.1, y0=0.5, x1=0.2, y1=0.55
            )
        )
        token_b = self.__token(
            layout=self.__row_layout(
                start=6, end=11, confidence=0.9, x0=0.2, y0=0.5, x1=0.3, y1=0.55
            )
        )
        single_line = self.__token(
            layout=self.__row_layout(
                start=0, end=11, confidence=0.9, x0=0.1, y0=0.5, x1=0.3, y1=0.55
            )
        )
        paragraph = self.__token(
            layout=self.__row_layout(
                start=0, end=11, confidence=0.9, x0=0.1, y0=0.5, x1=0.3, y1=0.55
            )
        )
        document = self.__document(
            text="Buy tickets",
            lines=(single_line,),
            paragraphs=(paragraph,),
            tokens=(token_a, token_b),
        )

        mapped = DocumentAiMapper().map_document(document=document, width=1000, height=2000)

        paragraphs = [element for element in mapped if element.level is OcrLevel.PARAGRAPH]
        self.assertEqual(paragraphs, [])

    def test_multi_segment_text_concatenates_in_order(self) -> None:
        """
        A layout whose text anchor carries multiple segments concatenates their slices in order.
        """

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
