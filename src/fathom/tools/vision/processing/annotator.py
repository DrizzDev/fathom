from __future__ import annotations

from logging import getLogger
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

try:
    from PIL.ImageFont import FreeTypeFont
except ImportError:
    FreeTypeFont = Any

from fathom.schemas.ui import LabeledElement
from fathom.tools.vision.processing.geometry import GeometryUtils

logger = getLogger(__name__)
BoundsTuple = Tuple[float, float, float, float]


class ImageAnnotator:
    """
    This class annotates images with bounding boxes and labels.
    """

    @classmethod
    def __get_text_size(
        cls, draw: ImageDraw.ImageDraw, text: str, font: FreeTypeFont
    ) -> Tuple[int, int]:
        """
        Gets the width and height of a text string for a given font.
        """
        try:
            bbox = draw.textbbox((0, 0), text, font=font)
            if isinstance(bbox, (tuple, list)) and len(bbox) == 4:
                return max(1, bbox[2] - bbox[0]), max(1, bbox[3] - bbox[1])
            else:
                return 10, 10  # Fallback
        except Exception:
            return 10, 10

    @classmethod
    def __load_fonts(
        cls, font_name: str, default_size: int, min_size: int, step: int = 2
    ) -> Dict[int, FreeTypeFont]:
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
        font_cache: Dict[int, FreeTypeFont],
    ) -> Optional[Tuple[FreeTypeFont, int, int]]:
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
        font: FreeTypeFont,
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
        font: FreeTypeFont,
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
    def annotate(
        cls,
        image_path: str,
        output_path: str,
        elements: List[LabeledElement],
        **kwargs: Any,
    ) -> Optional[str]:
        """
        Annotate an image with bounding boxes and labels.
        """
        font_name = "arial.ttf"
        font_size = int(kwargs.get("font_size", 24))
        min_font_size = int(kwargs.get("min_font_size", 10))
        box_thickness = int(kwargs.get("box_thickness", 2))

        image = None
        try:
            image = Image.open(image_path).convert("RGBA")
            draw = ImageDraw.Draw(image, "RGBA")

            font_cache = cls.__load_fonts(font_name, int(font_size), int(min_font_size))
            default_font = font_cache.get(int(font_size)) or list(font_cache.values())[0]
            sorted_font_sizes = sorted(font_cache.keys(), reverse=True)

            placed_outside_label_boxes: List[BoundsTuple] = []

            for element in elements:
                bounds = (
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
                final_label_box = None
                best_position = None

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
                    label=element.label,
                    color=element.color,
                    font=final_font,
                    draw=draw,
                    draw_connector_line=draw_connector_line,
                    element_bounds=bounds,
                    best_position=best_position,
                    final_label_box=final_label_box,
                )

            out_path = Path(output_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(output_path)
            logger.info(f"Saved annotated image: {output_path}")
            return output_path

        except Exception as exception:
            logger.exception(f"Failed to annotate: {exception}")
            return None
