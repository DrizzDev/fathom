from __future__ import annotations

from logging import getLogger
from typing import Any, List, Optional, Tuple

from fathom.constants.ocr import OcrConfidence, OcrLevel
from fathom.constants.perception import (
    OCR_CONFIDENCE_HIGH_FLOOR,
    OCR_CONFIDENCE_MEDIUM_FLOOR,
)
from fathom.schemas.actions import Bounds, CoordinateSource, CoordinateSystem
from fathom.schemas.ocr import OcrToken

logger = getLogger(__name__)


class DocumentAiMapper:
    """
    Translates Document AI response objects into typed OCR elements at
    three layout hierarchy levels:

    - :attr:`OcrLevel.TOKEN` — every word.
    - :attr:`OcrLevel.PARAGRAPH` — multi-line semantic blocks (e.g. an event-card title spanning multiple rows).
    - :attr:`OcrLevel.LINE` — multi-token rows merged by Document AI's own layout analysis (e.g. ``"Buy tickets from $ 0.00"``).

    Single-token lines and single-line paragraphs are suppressed at the source because they duplicate a lower-level element's bounds.
    """

    def map_document(self, *, document: Any, width: int, height: int) -> Tuple[OcrToken, ...]:
        """
        Convert every Document AI page into a flat tuple of typed
        :class:`OcrToken` instances spanning all three hierarchy levels.
        """

        elements: List[OcrToken] = []
        document_text = getattr(document, "text", "") or ""

        for page in document.pages:
            self.__log_response(page=page)

            line_spans = self.__collect_spans(items=getattr(page, "lines", ()) or ())
            token_spans = self.__collect_spans(items=getattr(page, "tokens", ()) or ())

            elements.extend(
                self.__map_level(
                    width=width,
                    height=height,
                    level=OcrLevel.TOKEN,
                    document_text=document_text,
                    items=getattr(page, "tokens", ()) or (),
                )
            )
            elements.extend(
                self.__map_level(
                    width=width,
                    height=height,
                    level=OcrLevel.LINE,
                    document_text=document_text,
                    contained_spans=token_spans,
                    items=getattr(page, "lines", ()) or (),
                )
            )
            elements.extend(
                self.__map_level(
                    width=width,
                    height=height,
                    level=OcrLevel.PARAGRAPH,
                    contained_spans=line_spans,
                    document_text=document_text,
                    items=getattr(page, "paragraphs", ()) or (),
                )
            )

        return tuple(elements)

    @staticmethod
    def __log_response(*, page: Any) -> None:
        """
        Emit per-page counts for every Document AI layout hierarchy level.
        These counts answer whether paragraphs / lines / blocks carry merge structure the mapper is currently discarding.
        """

        logger.info(
            "Document AI page layout levels",
            extra={
                "component": "adapter.ocr.document.mapper",
                "event": "ocr.document.layout.levels",
                "lines": getattr(page, "lines", ()) or (),
                "tokens": getattr(page, "tokens", ()) or (),
                "blocks": getattr(page, "blocks", ()) or (),
                "paragraphs": getattr(page, "paragraphs", ()) or (),
            },
        )

    def __map_level(
        self,
        *,
        width: int,
        height: int,
        level: OcrLevel,
        document_text: str,
        items: Tuple[Any, ...],
        contained_spans: Optional[Tuple[Tuple[int, int], ...]] = None,
    ) -> List[OcrToken]:
        """
        Map every item at one hierarchy level into typed elements,
        skipping any whose character span covers at most one ``contained_spans`` entry (single-child merges add no signal).
        """

        emitted: List[OcrToken] = []

        for item in items:
            layout = getattr(item, "layout", None)
            if layout is None:
                continue

            if contained_spans is not None and not self.__has_multiple_children(
                candidates=contained_spans,
                outer=self.__primary_span(layout=layout),
            ):
                continue

            if (
                element := self.__map_element(
                    level=level,
                    width=width,
                    height=height,
                    layout=layout,
                    document_text=document_text,
                )
            ) is not None:
                emitted.append(element)

        return emitted

    def __map_element(
        self,
        *,
        width: int,
        height: int,
        layout: Any,
        level: OcrLevel,
        document_text: str,
    ) -> Optional[OcrToken]:
        """
        Map one Document AI layout into a typed :class:`OcrToken` at the supplied level.
        """

        if not (snippet := self.__resolve_text(layout=layout, document_text=document_text)):
            return None

        if (bounds := self.__resolve_bounds(layout=layout, width=width, height=height)) is None:
            return None

        raw_score = float(getattr(layout, "confidence", 0.0) or 0.0)

        return OcrToken(
            level=level,
            text=snippet,
            bounds=bounds,
            raw_score=raw_score,
            confidence=self.__resolve_band(raw_score=raw_score),
        )

    @classmethod
    def __collect_spans(cls, *, items: Tuple[Any, ...]) -> Tuple[Tuple[int, int], ...]:
        """
        Collect the character span of every item's primary text segment.
        """

        spans: List[Tuple[int, int]] = []

        for item in items:
            layout = getattr(item, "layout", None)
            if layout is None:
                continue

            if (span := cls.__primary_span(layout=layout)) is not None:
                spans.append(span)

        return tuple(spans)

    @staticmethod
    def __primary_span(*, layout: Any) -> Optional[Tuple[int, int]]:
        """
        Return the first ``(start, end)`` segment span of the layout, when present.
        """

        anchor = getattr(layout, "text_anchor", None)

        if anchor is None:
            return None

        for segment in getattr(anchor, "text_segments", None) or ():
            start = int(getattr(segment, "start_index", 0) or 0)
            end = int(getattr(segment, "end_index", 0) or 0)

            if end > start:
                return start, end

        return None

    @staticmethod
    def __has_multiple_children(
        *,
        outer: Optional[Tuple[int, int]],
        candidates: Tuple[Tuple[int, int], ...],
    ) -> bool:
        """
        Return whether ``candidates`` contains at least two spans inside ``outer``.
        """

        if outer is None:
            return False

        count = 0
        outer_start, outer_end = outer

        for child_start, child_end in candidates:
            if outer_start <= child_start and child_end <= outer_end:
                count += 1
                if count > 1:
                    return True

        return False

    def __resolve_text(self, *, layout: Any, document_text: str) -> str:
        """
        Resolve a token's surface text from the layout text anchor.
        """

        if (anchor := getattr(layout, "text_anchor", None)) is None:
            return ""

        if not (segments := getattr(anchor, "text_segments", None) or ()):
            return ""

        parts: List[str] = []

        for segment in segments:
            end = int(getattr(segment, "end_index", 0) or 0)
            start = int(getattr(segment, "start_index", 0) or 0)

            parts.append(document_text[start:end])

        return "".join(parts).strip()

    def __resolve_bounds(self, *, layout: Any, width: int, height: int) -> Optional[Bounds]:
        """
        Build pixel-space :class:`Bounds` from a Document AI layout polygon.
        """

        if (polygon := getattr(layout, "bounding_poly", None)) is None:
            return None

        if vertices := getattr(polygon, "normalized_vertices", None) or ():
            scale_x, scale_y = float(width), float(height)

        elif vertices := getattr(polygon, "vertices", None) or ():
            scale_x, scale_y = 1.0, 1.0

        else:
            return None

        xs = [float(getattr(vertex, "x", 0.0) or 0.0) for vertex in vertices]
        ys = [float(getattr(vertex, "y", 0.0) or 0.0) for vertex in vertices]

        # Some Document AI preview processors emit absolute vertices outside
        # the image rectangle; clamp both axes uniformly so downstream pixel bounds never escape the screen.
        max_x = float(width)
        max_y = float(height)

        x_min = max(0.0, min(min(xs) * scale_x, max_x))
        y_min = max(0.0, min(min(ys) * scale_y, max_y))
        x_max = max(0.0, min(max(xs) * scale_x, max_x))
        y_max = max(0.0, min(max(ys) * scale_y, max_y))

        bound_width = max(0, int(x_max - x_min))
        bound_height = max(0, int(y_max - y_min))

        if bound_width == 0 or bound_height == 0:
            return None

        return Bounds(
            x=int(x_min),
            y=int(y_min),
            width=bound_width,
            height=bound_height,
            source=CoordinateSource.OCR,
            coordinate_system=CoordinateSystem.DEVICE_PIXEL,
        )

    def __resolve_band(self, *, raw_score: float) -> OcrConfidence:
        """
        Map a numeric provider score onto the coarse OCR confidence band.
        """

        if raw_score >= OCR_CONFIDENCE_HIGH_FLOOR:
            return OcrConfidence.HIGH

        if raw_score >= OCR_CONFIDENCE_MEDIUM_FLOOR:
            return OcrConfidence.MEDIUM

        return OcrConfidence.LOW
