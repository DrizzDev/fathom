from __future__ import annotations

from pathlib import Path
from typing import Final, Mapping, Optional, Tuple

from PIL import ImageDraw, ImageFont

from fathom.constants.drawing import BoxDrawing, SourceColor
from fathom.schemas.observation import ElementSource


class BoxDrawer:
    """
    Single shared primitive for drawing labelled boxes on annotated artifacts.

    Owns font, font size, outline width, label format, and the source-keyed colour palette so every
    annotated-artifact renderer draws in a consistent style. Label priority is numeric ``label_id``
    first, then visible text, then role — matching the numeric labels the planner reads off the manifest.
    """

    __SOURCE_COLOURS: Final[Mapping[ElementSource, str]] = {
        ElementSource.XML: SourceColor.XML,
        ElementSource.OCR: SourceColor.OCR,
        ElementSource.CV: SourceColor.CV,
        ElementSource.ICON: SourceColor.ICON,
        ElementSource.MODEL: SourceColor.MODEL,
        ElementSource.VISION: SourceColor.VISION,
        ElementSource.ACCESSIBILITY: SourceColor.ACCESSIBILITY,
    }

    __FONT_SEARCH_PATHS: Final[Tuple[str, ...]] = (
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )

    def __init__(
        self,
        *,
        line_width: int = BoxDrawing.LINE_WIDTH,
        font_size: int = BoxDrawing.FONT_SIZE_DEFAULT,
    ) -> None:
        """
        Bind the drawer to its stroke width and label font size.
        """

        self.__line_width = line_width
        self.__font_size = font_size
        self.__font = self.__resolve_font(size=font_size)

    def draw(
        self,
        *,
        canvas: ImageDraw.ImageDraw,
        bounds: Tuple[int, int, int, int],
        source: ElementSource,
        label_id: Optional[str] = None,
        text: Optional[str] = None,
        role: Optional[str] = None,
        color: Optional[str] = None,
    ) -> None:
        """
        Render one outlined rectangle and its label.

        ``color`` overrides the source-keyed palette when set; used by
        callers that paint non-element overlays (call-to-action,
        overlays, trace arrows) and want a deterministic colour.
        """

        outline = color or self.__SOURCE_COLOURS.get(source, SourceColor.FALLBACK)
        canvas.rectangle(bounds, outline=outline, width=self.__line_width)
        canvas.text(
            self.__label_position(bounds=bounds),
            self.__compose_label(label_id=label_id, text=text, role=role),
            font=self.__font,
            fill=outline,
            stroke_width=BoxDrawing.LABEL_STROKE_WIDTH,
            stroke_fill=BoxDrawing.LABEL_STROKE_COLOR,
        )

    def color_for(self, *, source: ElementSource) -> str:
        """
        Return the canonical colour for one perception source.
        """

        return self.__SOURCE_COLOURS.get(source, SourceColor.FALLBACK)

    @staticmethod
    def __compose_label(
        *,
        label_id: Optional[str],
        text: Optional[str],
        role: Optional[str],
    ) -> str:
        """
        Pick the most informative label: numeric ``label_id`` (matches the manifest the LLM sees),
        else visible text, else role as a last-resort marker so empty boxes never go un-labelled.
        """

        if label_id:
            return label_id

        if text and text.strip():
            return text.strip()[:32]

        if role:
            return role

        return ""

    @staticmethod
    def __label_position(*, bounds: Tuple[int, int, int, int]) -> Tuple[int, int]:
        """
        Anchor the label at the top-left interior of the box; drawing inside keeps labels readable on
        dense screens where stacked boxes make outside labels overlap.
        """

        x1, y1, _x2, _y2 = bounds
        return (
            x1 + BoxDrawing.LABEL_PADDING,
            y1 + BoxDrawing.LABEL_PADDING,
        )

    @classmethod
    def __resolve_font(cls, *, size: int) -> ImageFont.ImageFont:
        """
        Load a TrueType font at the requested size, falling back to the Pillow default
        (``ImageFont.load_default()``) so rendering never crashes when no system font path exists.
        """

        for path in cls.__FONT_SEARCH_PATHS:
            if Path(path).exists():
                try:
                    return ImageFont.truetype(path, size)
                except OSError:
                    continue

        return ImageFont.load_default()
