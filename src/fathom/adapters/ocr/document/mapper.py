from __future__ import annotations

from typing import Any, List, Optional, Tuple

from fathom.constants.perception import (
    OCR_CONFIDENCE_HIGH_FLOOR,
    OCR_CONFIDENCE_MEDIUM_FLOOR,
)
from fathom.schemas.actions import Bounds, CoordinateSource, CoordinateSystem
from fathom.schemas.ocr import OcrConfidence, OcrToken


class DocumentAiMapper:
    """
    Translates Document AI response objects into typed OCR tokens.
    """

    def map_document(self, *, document: Any, width: int, height: int) -> Tuple[OcrToken, ...]:
        """
        Convert every Document AI page token into a typed :class:`OcrToken`.
        """

        tokens: List[OcrToken] = []
        document_text = getattr(document, "text", "") or ""

        for page in document.pages:
            for raw_token in page.tokens:
                if (
                    mapped := self.__map_token(
                        width=width,
                        height=height,
                        token=raw_token,
                        document_text=document_text,
                    )
                ) is not None:
                    tokens.append(mapped)
        return tuple(tokens)

    def __map_token(
        self,
        *,
        token: Any,
        width: int,
        height: int,
        document_text: str,
    ) -> Optional[OcrToken]:
        """
        Map one Document AI page token into a typed OCR token.
        """

        if not (snippet := self.__resolve_text(layout=token.layout, document_text=document_text)):
            return None

        if (
            bounds := self.__resolve_bounds(layout=token.layout, width=width, height=height)
        ) is None:
            return None

        raw_score = float(getattr(token.layout, "confidence", 0.0) or 0.0)

        return OcrToken(
            text=snippet,
            bounds=bounds,
            raw_score=raw_score,
            confidence=self.__resolve_band(raw_score=raw_score),
        )

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
            start = int(getattr(segment, "start_index", 0) or 0)
            end = int(getattr(segment, "end_index", 0) or 0)
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
        # the image rectangle; clamp both axes uniformly so downstream pixel
        # bounds never escape the screen.
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
