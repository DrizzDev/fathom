import xml.etree.ElementTree as ET  # nosec
from logging import getLogger
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from fathom.constants import ActionType
from fathom.schemas.ui import LabeledElement, UIBoundingBox
from fathom.tools.vision.processing.parsers.factory import PlatformParserFactory

logger = getLogger(__name__)


class BoundingBoxGenerator:
    """
    This class is responsible for generating bounding boxes for elements.
    """

    _COLORS = [
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
        xml_root: ET.Element,
        screenshot_path: str,
        action: Optional[ActionType] = None,
        **kwargs: Any,
    ) -> Tuple[List[LabeledElement], Dict[str, Any]]:
        """
        Create bounding boxes for elements.
        """

        logger.info(
            f"BoundingBoxGenerator.create_element called with screenshot_path: {screenshot_path}, action: {action}"
        )

        parser = PlatformParserFactory.get_parser(xml_root)
        if not parser:
            logger.error(f"No parser found for XML root tag: {xml_root.tag}")
            return [], {}

        logger.info(f"Using parser: {type(parser).__name__}")

        try:
            with Image.open(screenshot_path) as img:
                screenshot_width = img.width
                screenshot_height = img.height
                logger.info(
                    f"Screenshot opened successfully: {screenshot_width}x{screenshot_height}"
                )

        except FileNotFoundError:
            logger.error(f"Screenshot not found at {screenshot_path}")
            return [], {}

        except Exception as exception:
            logger.error(f"Failed to open screenshot at {screenshot_path}: {exception}")
            return [], {}

        scale_factor = parser.get_scale_factor(xml_root, screenshot_width)
        logger.info(f"Scale factor calculated: {scale_factor}")

        logical_elements = parser.find_all_elements(
            xml_root,
            screenshot_width=screenshot_width,
            screenshot_height=screenshot_height,
            **kwargs,
        )
        logger.info(f"Raw elements found: {len(logical_elements)}")

        if action:
            logger.info(f"Applying pre-filter for action: '{action}'")
            logical_elements = parser.filter_by_action(logical_elements, action)
            logger.info(f"After pre-filter for action: {action}: {len(logical_elements)}")

        filtered_elements = parser.filter_and_deduplicate(logical_elements, action=action)
        logger.info(f"After deduplication filter: {len(filtered_elements)}")

        label_map: Dict[str, Any] = {}
        final_labeled_elements = []
        logger.info(f"Final element count after all filtering: {len(filtered_elements)}")

        for index, element in enumerate(filtered_elements, start=1):
            # Keep original logical bounds for label map (used for tapping)
            logical_bounds = element.bounds

            # Create scaled bounds for drawing on high-res screenshot
            scaled_bounds = UIBoundingBox(
                x1=logical_bounds.x1 * scale_factor,
                y1=logical_bounds.y1 * scale_factor,
                x2=logical_bounds.x2 * scale_factor,
                y2=logical_bounds.y2 * scale_factor,
            )

            # Use simple numeric labels for precision LLM grounding
            label = str(index)

            # Set element properties
            element.label = label
            element.bounds = scaled_bounds  # Use scaled bounds for drawing
            element.color = cls._COLORS[(index - 1) % len(cls._COLORS)]

            # Store logical bounds in attributes for later use
            element.attributes["logical_bounds"] = {
                "x1": logical_bounds.x1,
                "y1": logical_bounds.y1,
                "x2": logical_bounds.x2,
                "y2": logical_bounds.y2,
            }

            final_labeled_elements.append(element)

            # Use logical coordinates in label map for all touch actions
            bounds = f"[{int(logical_bounds.x1)},{int(logical_bounds.y1)}][{int(logical_bounds.x2)},{int(logical_bounds.y2)}]"

            label_map[element.label] = {
                **{
                    key: value
                    for key, value in element.attributes.items()
                    if key != "logical_bounds"
                },
                "bounds": bounds,
                "center_x": int((logical_bounds.x1 + logical_bounds.x2) / 2),
                "center_y": int((logical_bounds.y1 + logical_bounds.y2) / 2),
            }
            logger.debug(
                f"Element Label {element.label}: {element.attributes.get('class')} - text='{element.attributes.get('text')}'"
            )

        label_map["__scale_factor__"] = float(scale_factor)
        logger.info(f"Successfully generated {len(final_labeled_elements)} labeled elements.")

        return final_labeled_elements, label_map
