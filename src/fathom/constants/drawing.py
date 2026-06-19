from __future__ import annotations

from typing import Final


class BoxDrawing:
    """
    Geometry constants shared across every annotated-artifact renderer.

    Centralised so the XML manifest annotator, the merged perception
    renderer, and the per-source perception renderers cannot drift apart
    on stroke width, font sizing, padding, or label positioning. The
    values mirror :class:`fathom.processing.annotator.ImageAnnotator`'s
    original defaults (font 24, stroke 2 outlined white, line width 2)
    so any image stitched together from multiple sources reads as a
    single coherent annotation pass instead of a multi-style collage.
    """

    LINE_WIDTH: Final[int] = 2
    LABEL_PADDING: Final[int] = 4
    LABEL_STROKE_WIDTH: Final[int] = 2
    LABEL_STROKE_COLOR: Final[str] = "white"
    FONT_SIZE_DEFAULT: Final[int] = 24
    FONT_SIZE_MINIMUM: Final[int] = 10
    FONT_SIZE_STEP: Final[int] = 2


class TraceDrawing:
    """
    Geometry constants for the action-trace renderer (tap circles, swipe arrows).
    """

    LINE_WIDTH: Final[int] = 10
    TAP_RADIUS: Final[int] = 40
    CENTER_DOT_RADIUS: Final[int] = 5
    SWIPE_START_RADIUS: Final[int] = 15
    ARROW_HEAD_LENGTH: Final[int] = 24
    ARROW_HEAD_ANGLE_DEGREES: Final[int] = 30


class SourceColor:
    """
    Source-keyed colour palette shared across annotated-artifact
    renderers. Stable so reviewers can read element source from colour
    without consulting a legend.
    """

    XML: Final[str] = "#3B82F6"
    OCR: Final[str] = "#10B981"
    CV: Final[str] = "#A855F7"
    ICON: Final[str] = "#F59E0B"
    MODEL: Final[str] = "#EAB308"
    VISION: Final[str] = "#EC4899"
    ACCESSIBILITY: Final[str] = "#8B5CF6"
    OVERLAY: Final[str] = "#EF4444"
    CALL_TO_ACTION: Final[str] = "#F97316"
    FALLBACK: Final[str] = "#9CA3AF"
    ACTION: Final[str] = "#FF3B30"
