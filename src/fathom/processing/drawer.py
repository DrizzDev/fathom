from __future__ import annotations

from logging import getLogger
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    import xml.etree.ElementTree as ET  # nosec

from PIL import Image

from fathom.constants import ActionType
from fathom.processing.parsers.factory import PlatformParserFactory
from fathom.schemas.ui import LabeledElement, UIBounds

logger = getLogger(name=__name__)


class BoundsGenerator:
    """
    This class is responsible for generating visual bounds for UI elements.
    """

    __COLORS = [
        "#FF3B30",
        "#34C759",
        "#007AFF",
        "#FF9500",
        "#AF52DE",
        "#5856D6",
        "#FF2D55",
        "#5AC8FA",
        "#FFCC00",
        "#FF6482",
        "#8E8E93",
        "#32ADE6",
    ]

    @classmethod
    def create_element(
        cls,
        root: ET.Element,
        image_path: str,
        action: Optional[ActionType] = None,
        **extra: Any,
    ) -> Tuple[List[LabeledElement], Dict[str, Any]]:
        """
        Create bounding boxes for elements.
        """

        logger.info(
            f"BoundsGenerator.create_element called with image_path: {image_path}, action: {action}"
        )

        parser = PlatformParserFactory.get_parser(root=root)

        logger.info(f"Using parser: {type(parser).__name__}")

        try:
            with Image.open(fp=image_path) as image:
                width = image.width
                height = image.height
                logger.info(f"Screenshot opened successfully: {width}x{height}")

        except FileNotFoundError:
            logger.error(f"Screenshot not found at {image_path}")
            return [], {}

        except Exception as exception:
            logger.error(f"Failed to open screenshot at {image_path}: {exception}")
            return [], {}

        factor = parser.get_scale_factor(root=root, screenshot_width=width)
        logger.info(f"Scale factor calculated: {factor}")

        logical = parser.find_all_elements(
            root=root,
            screenshot_width=width,
            screenshot_height=height,
            **extra,
        )
        logger.info(f"Raw elements found: {len(logical)}")

        if action:
            logger.info(f"Applying pre-filter for action: '{action}'")
            logical = parser.filter_by_action(elements=logical, action=action)
            logger.info(f"After pre-filter for action: {action}: {len(logical)}")

        filtered = parser.filter_and_deduplicate(elements=logical, action=action)
        logger.info(f"After deduplication filter: {len(filtered)}")

        mapping: Dict[str, Any] = {}
        labeled = []
        logger.info(f"Final element count after all filtering: {len(filtered)}")

        for index, element in enumerate(iterable=filtered, start=1):
            # Keep original logical bounds for platform-specific metadata.
            logic = element.bounds

            # Create scaled bounds for drawing on high-res screenshot
            scaled = UIBounds(
                x1=logic.x1 * factor,
                y1=logic.y1 * factor,
                x2=logic.x2 * factor,
                y2=logic.y2 * factor,
            )

            # Use simple numeric labels for precision LLM grounding
            label = str(index)

            # Set element properties
            element.label = label
            element.bounds = scaled  # Use scaled bounds for drawing
            element.color = cls.__COLORS[(index - 1) % len(cls.__COLORS)]

            # Store logical bounds in attributes for later use
            element.attributes["logical_bounds"] = {
                "x1": logic.x1,
                "y1": logic.y1,
                "x2": logic.x2,
                "y2": logic.y2,
            }

            labeled.append(element)

            # Use rendered screenshot coordinates in the label map for action execution.
            serialization = (
                f"[{int(round(scaled.x1))},{int(round(scaled.y1))}]"
                f"[{int(round(scaled.x2))},{int(round(scaled.y2))}]"
            )

            mapping[element.label] = {
                **{
                    key: value
                    for key, value in element.attributes.items()
                    if key != "logical_bounds"
                },
                "bounds": serialization,
                "center_x": int(round((scaled.x1 + scaled.x2) / 2.0)),
                "center_y": int(round((scaled.y1 + scaled.y2) / 2.0)),
                "logical_bounds": {
                    "x1": int(logic.x1),
                    "y1": int(logic.y1),
                    "x2": int(logic.x2),
                    "y2": int(logic.y2),
                },
            }

        mapping["__scale_factor__"] = float(factor)
        logger.info(f"Successfully generated {len(labeled)} labeled elements.")

        return labeled, mapping
