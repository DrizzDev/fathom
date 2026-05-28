from __future__ import annotations

import io
from logging import getLogger
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, TypeAlias, Union

from PIL import Image, ImageDraw, ImageFont

if TYPE_CHECKING:
    from PIL.ImageFont import FreeTypeFont
    from PIL.ImageFont import ImageFont as PILImageFont

    FontType: TypeAlias = Union[FreeTypeFont, PILImageFont]
else:
    try:
        from PIL.ImageFont import FreeTypeFont
    except ImportError:
        FreeTypeFont = Any

    FontType = Any

from fathom.constants.drawing import SourceColor
from fathom.processing.geometry import GeometryUtils
from fathom.schemas.observation import ElementSource
from fathom.schemas.ui import LabeledElement

logger = getLogger(__name__)
BoundsTuple = Tuple[float, float, float, float]


class ImageAnnotator:
    """
    This class annotates images with bounding boxes and labels.
    """

    @classmethod
    def __get_text_size(
        cls, draw: ImageDraw.ImageDraw, text: str, font: FontType
    ) -> Tuple[int, int]:
        """
        Gets the width and height of a text string for a given font.
        """

        try:
            bbox = draw.textbbox((0, 0), text, font=font)
            if isinstance(bbox, (tuple, list)) and len(bbox) == 4:
                width = max(1, int(bbox[2] - bbox[0]))
                height = max(1, int(bbox[3] - bbox[1]))
                return width, height
            else:
                return 10, 10  # Fallback
        except Exception:
            return 10, 10

    @classmethod
    def __load_fonts(
        cls, font_name: str, default_size: int, min_size: int, step: int = 2
    ) -> Dict[int, FontType]:
        """
        Pre-loads all required font sizes into a cache once.
        """

        cache = {}
        step = abs(step) if step != 0 else 2
        requested_sizes = set(range(default_size, min_size - 1, -step)) | {
            min_size,
            default_size,
        }
        valid_sizes = {size for size in requested_sizes if size >= 1}

        # Try system fonts if custom font not found
        font_paths = [
            font_name,
            "/Library/Fonts/Arial.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]

        found_font = None
        for path in font_paths:
            if Path(path).exists():
                found_font = path
                break

        try:
            if found_font:
                for size in sorted(valid_sizes, reverse=True):
                    if size not in cache:
                        cache[size] = ImageFont.truetype(found_font, size)
            else:
                # Fallback to default
                default_font = ImageFont.load_default()
                cache = dict.fromkeys(valid_sizes, default_font)

        except Exception as exception:
            logger.warning(f"Error loading fonts: {exception}")
            default_font = ImageFont.load_default()
            cache = dict.fromkeys(valid_sizes, default_font)

        return cache

    @classmethod
    def __find_best_font_for_inside(
        cls,
        label: str,
        width: float,
        height: float,
        padding: int,
        draw: ImageDraw.ImageDraw,
        sorted_font_sizes: List[int],
        font_cache: Dict[int, FontType],
    ) -> Optional[Tuple[FontType, int, int]]:
        """
        Finds the largest font from the cache that fits the label inside the box.
        """

        if width <= 0 or height <= 0:
            return None

        for size in sorted_font_sizes:
            font = font_cache.get(size)
            if not font:
                continue

            text_width, text_height = cls.__get_text_size(draw, label, font)
            if (text_width + padding <= width) and (text_height + padding <= height):
                return font, text_width, text_height

        return None

    @classmethod
    def __get_outside_position(
        cls,
        label: str,
        image_width: int,
        image_height: int,
        font: Any,
        bounds: BoundsTuple,
        draw: ImageDraw.ImageDraw,
        _placed_label_boxes: List[BoundsTuple],
    ) -> Tuple[Tuple[float, float], BoundsTuple]:
        """
        Calculates the best 'outside' position.
        """

        padding = 8
        text_width, text_height = cls.__get_text_size(draw, label, font)

        candidates = [
            (bounds[0], bounds[1] - text_height - padding),
            (bounds[2] - text_width, bounds[1] - text_height - padding),
            (bounds[0], bounds[3] + padding),
            (bounds[2] - text_width, bounds[3] + padding),
            (
                bounds[0] + (bounds[2] - bounds[0]) / 2 - text_width / 2,
                bounds[1] - text_height - padding,
            ),
            (
                bounds[0] + (bounds[2] - bounds[0]) / 2 - text_width / 2,
                bounds[3] + padding,
            ),
        ]
        best_position = candidates[4]  # Default to centered above

        for position in candidates:
            x, y = position
            x = max(0, min(x, image_width - text_width))
            y = max(0, min(y, image_height - text_height))

            best_position = (x, y)
            break

        final_label_box: BoundsTuple = (
            best_position[0],
            best_position[1],
            best_position[0] + text_width,
            best_position[1] + text_height,
        )

        return best_position, final_label_box

    @classmethod
    def __draw_label_and_connector(
        cls,
        label: str,
        color: str,
        font: Any,
        draw: ImageDraw.ImageDraw,
        draw_connector_line: bool,
        element_bounds: BoundsTuple,
        best_position: Tuple[float, float],
        final_label_box: Optional[BoundsTuple],
    ) -> None:
        """
        Draws the label and connector.
        """

        if draw_connector_line and final_label_box:
            try:
                point_on_label, point_on_box = GeometryUtils.get_line_endpoints(
                    final_label_box, element_bounds
                )
                draw.line([point_on_label, point_on_box], fill="grey", width=1)
            except Exception:
                pass  # nosec

        draw.text(
            xy=best_position,
            text=label,
            font=font,
            fill=color,
            stroke_width=2,
            stroke_fill="white",
        )

    @classmethod
    def trace(
        cls,
        image_data: bytes,
        output_path: str,
        action_type: str,
        coords: Tuple[int, ...],
        label: str = "",
    ) -> Optional[str]:
        """
        Draw action indicator on image for background verification.

        .. deprecated::
            Use :class:`fathom.core.artifact.pipeline.ArtifactPipeline`
            with :class:`fathom.schemas.artifact.TracePayload` instead.
            This direct-write path is retained as a fallback for
            ad-hoc tooling that has not migrated yet.
        """

        import warnings

        warnings.warn(
            "ImageAnnotator.trace is deprecated; emit a TracePayload via ArtifactPipeline instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        try:
            import io

            image = Image.open(io.BytesIO(image_data)).convert("RGB")
            draw = ImageDraw.Draw(image, "RGBA")

            # Use red/orange for visibility
            color = "#FF3B30"
            alpha_fill = (255, 59, 48, 100)  # Semi-transparent

            if action_type in ("tap", "type", "long_press"):
                if len(coords) >= 2:
                    x, y = coords[0], coords[1]
                    # Draw a target circle
                    r = 40
                    draw.ellipse(
                        [x - r, y - r, x + r, y + r], outline=color, width=5, fill=alpha_fill
                    )
                    draw.ellipse([x - 5, y - 5, x + 5, y + 5], fill=color)  # Center dot

            elif (
                action_type
                in (
                    "swipe",
                    "scroll",
                    "swipe_left",
                    "swipe_right",
                    "swipe_up",
                    "swipe_down",
                )
                and len(coords) >= 4
            ):
                x1, y1, x2, y2 = coords[0], coords[1], coords[2], coords[3]
                # Draw arrow line
                draw.line([x1, y1, x2, y2], fill=color, width=10)
                # Simple arrowhead (circle at start, cross at end)
                draw.ellipse([x1 - 15, y1 - 15, x1 + 15, y1 + 15], fill=color)
                draw.line([x2 - 20, y2 - 20, x2 + 20, y2 + 20], fill=color, width=10)
                draw.line([x2 + 20, y2 - 20, x2 - 20, y2 + 20], fill=color, width=10)

            # Add label if provided
            if label:
                font = ImageFont.load_default()
                draw.text(
                    (10, 10),
                    f"Action: {label}",
                    font=font,
                    fill="white",
                    stroke_width=2,
                    stroke_fill="black",
                )

            out_path = Path(output_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(output_path)
            return str(out_path)

        except Exception as exception:
            logger.warning(f"Verification annotation failed: {exception}")
            return None

    @classmethod
    def annotate(
        cls,
        *,
        image: bytes,
        elements: List[LabeledElement],
        font_size: int = 24,
        min_font_size: int = 10,
        box_thickness: int = 2,
    ) -> Optional[bytes]:
        """
        Render bounding boxes and labels onto in-memory image bytes.

        Pure transform — bytes-in, bytes-out — so the annotation stage
        owns no filesystem state and stays independent of any artifact
        staging lifecycle. Returns ``None`` only when the input bytes
        cannot be decoded; emit failures are logged with context.
        """

        if not image:
            raise ValueError("ImageAnnotator.annotate requires non-empty image bytes")

        font_name = "arial.ttf"

        try:
            with Image.open(io.BytesIO(image)) as source:
                canvas = source.convert("RGBA")

            draw = ImageDraw.Draw(canvas, "RGBA")

            font_cache = cls.__load_fonts(font_name, int(font_size), int(min_font_size))
            default_font = font_cache.get(int(font_size)) or list(font_cache.values())[0]
            sorted_font_sizes = sorted(font_cache.keys(), reverse=True)

            placed_outside_label_boxes: List[BoundsTuple] = []

            for element in elements:
                bounds: BoundsTuple = (
                    float(element.bounds.x1),
                    float(element.bounds.y1),
                    float(element.bounds.x2),
                    float(element.bounds.y2),
                )

                draw.rectangle(bounds, outline=element.color, width=box_thickness)

                padding_inside = box_thickness + 2
                box_width = bounds[2] - bounds[0]
                box_height = bounds[3] - bounds[1]

                final_font = default_font
                draw_connector_line = False
                final_label_box: Optional[BoundsTuple] = None
                best_position: Tuple[float, float]

                fit_result = cls.__find_best_font_for_inside(
                    label=element.label,
                    width=box_width,
                    height=box_height,
                    padding=padding_inside,
                    draw=draw,
                    sorted_font_sizes=sorted_font_sizes,
                    font_cache=font_cache,
                )

                if fit_result:
                    final_font, text_width, text_height = fit_result
                    best_position = (
                        bounds[0] + padding_inside,
                        bounds[1] + padding_inside,
                    )
                    final_label_box = (
                        best_position[0],
                        best_position[1],
                        best_position[0] + text_width,
                        best_position[1] + text_height,
                    )
                else:
                    best_position, final_label_box = cls.__get_outside_position(
                        label=element.label,
                        image_width=canvas.width,
                        image_height=canvas.height,
                        font=default_font,
                        bounds=bounds,
                        draw=draw,
                        _placed_label_boxes=placed_outside_label_boxes,
                    )
                    draw_connector_line = True
                    placed_outside_label_boxes.append(final_label_box)

                cls.__draw_label_and_connector(
                    label=element.label,
                    color=element.color,
                    font=final_font,
                    draw=draw,
                    draw_connector_line=draw_connector_line,
                    element_bounds=bounds,
                    best_position=best_position,
                    final_label_box=final_label_box,
                )

            buffer = io.BytesIO()
            canvas.convert("RGB").save(buffer, format="PNG")
            return buffer.getvalue()

        except Exception as exception:
            logger.exception(f"Failed to annotate: {exception}")
            return None

    __PERCEPTION_SOURCE_COLOURS: Dict[ElementSource, str] = {
        ElementSource.OCR: SourceColor.OCR,
        ElementSource.ICON: SourceColor.ICON,
        ElementSource.VISION: SourceColor.VISION,
    }

    @classmethod
    def overlay_perception_boxes(
        cls,
        *,
        image_bytes: bytes,
        entries: List[Any],
        font_size: int = 24,
        min_font_size: int = 10,
        box_thickness: int = 2,
    ) -> Optional[bytes]:
        """
        Draw perception-source bounding boxes on top of an existing
        annotated image and return the new PNG bytes.

        Used when :class:`fathom.core.services.manifest.ManifestMerger`
        appends OCR / Icon / Vision elements onto the XML manifest:
        the LLM-facing image must show a box for every numeric label
        the LLM sees in the manifest, otherwise the model is being
        asked to ground against labels it cannot visually verify
        (a documented hallucination regime).

        ``entries`` are :class:`AppendedManifestEntry` tuples carrying
        ``(label_id, source, text, bounds)`` — colours come from the
        :class:`SourceColor` palette so OCR boxes are green, Icon
        amber, Vision pink, distinguishable from the blue XML and
        purple CV boxes already on the image. Styling (font size,
        stroke width, label positioning) matches :meth:`annotate`
        exactly so the combined image reads as one consistent pass.
        """

        if not entries:
            return image_bytes

        import io

        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
            draw = ImageDraw.Draw(image, "RGBA")

            font_cache = cls.__load_fonts("arial.ttf", int(font_size), int(min_font_size))
            default_font = font_cache.get(int(font_size)) or next(iter(font_cache.values()))
            sorted_font_sizes = sorted(font_cache.keys(), reverse=True)

            placed_outside_label_boxes: List[BoundsTuple] = []

            for entry in entries:
                colour = cls.__PERCEPTION_SOURCE_COLOURS.get(entry.source, SourceColor.FALLBACK)
                bounds: BoundsTuple = (
                    float(entry.bounds[0]),
                    float(entry.bounds[1]),
                    float(entry.bounds[2]),
                    float(entry.bounds[3]),
                )

                draw.rectangle(bounds, outline=colour, width=box_thickness)

                padding_inside = box_thickness + 2
                box_width = bounds[2] - bounds[0]
                box_height = bounds[3] - bounds[1]

                label = str(entry.label_id)
                draw_connector_line = False
                final_label_box: Optional[BoundsTuple] = None
                best_position: Tuple[float, float]

                fit_result = cls.__find_best_font_for_inside(
                    label=label,
                    width=box_width,
                    height=box_height,
                    padding=padding_inside,
                    draw=draw,
                    sorted_font_sizes=sorted_font_sizes,
                    font_cache=font_cache,
                )

                if fit_result:
                    final_font, text_width, text_height = fit_result
                    best_position = (
                        bounds[0] + padding_inside,
                        bounds[1] + padding_inside,
                    )
                    final_label_box = (
                        best_position[0],
                        best_position[1],
                        best_position[0] + text_width,
                        best_position[1] + text_height,
                    )
                else:
                    final_font = default_font
                    best_position, final_label_box = cls.__get_outside_position(
                        label=label,
                        image_width=image.width,
                        image_height=image.height,
                        font=default_font,
                        bounds=bounds,
                        draw=draw,
                        _placed_label_boxes=placed_outside_label_boxes,
                    )
                    draw_connector_line = True
                    placed_outside_label_boxes.append(final_label_box)

                cls.__draw_label_and_connector(
                    label=label,
                    color=colour,
                    font=final_font,
                    draw=draw,
                    draw_connector_line=draw_connector_line,
                    element_bounds=bounds,
                    best_position=best_position,
                    final_label_box=final_label_box,
                )

            buffer = io.BytesIO()
            image.convert("RGB").save(buffer, format="PNG")
            return buffer.getvalue()

        except Exception as exception:
            logger.warning(f"Failed to overlay perception boxes: {exception}")
            return image_bytes
