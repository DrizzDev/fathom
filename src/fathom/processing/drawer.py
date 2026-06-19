from __future__ import annotations

from io import BytesIO
from logging import getLogger
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    import xml.etree.ElementTree as ET  # nosec

from PIL import Image, UnidentifiedImageError

from fathom.constants import ActionType
from fathom.processing.cv_labeler import VisualControlLabeler
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
        *,
        root: ET.Element,
        image: bytes,
        action: Optional[ActionType] = None,
        cv_enabled: bool = False,
        **extra: Any,
    ) -> Tuple[List[LabeledElement], Dict[str, Any]]:
        """
        Create bounding boxes for elements from in-memory screenshot bytes.

        Consumers pass the canonical PNG bytes carried by
        :class:`ScreenCapture` so this stage stays decoupled from any
        filesystem-staging artifact lifecycle owned by the artifact
        pipeline. ``cv_enabled`` defaults to False so the OpenCV
        :class:`VisualControlLabeler` is skipped on the original
        XML+LLM-only flow. When True, CV-detected regions are appended
        to the manifest as fallback anchors for icon-only screens.
        """

        if not image:
            raise ValueError("BoundsGenerator.create_element requires non-empty image bytes")

        logger.info(
            "BoundsGenerator.create_element invoked",
            extra={
                "component": "processing.drawer",
                "event": "drawer.create_element.invoked",
                "action": str(action) if action is not None else None,
                "image_bytes": len(image),
                "cv_enabled": cv_enabled,
            },
        )

        parser = PlatformParserFactory.get_parser(root=root)
        logger.info(f"Using parser: {type(parser).__name__}")

        try:
            with Image.open(fp=BytesIO(image)) as decoded:
                width = decoded.width
                height = decoded.height
                logger.info(f"Screenshot decoded successfully: {width}x{height}")

        except UnidentifiedImageError as exception:
            logger.error(f"Failed to decode screenshot bytes: {exception}")
            return [], {}

        factor = parser.get_scale_factor(root=root, screenshot_width=width)
        logger.info(f"Scale factor calculated: {factor}")

        logical = parser.find_all_elements(
            root=root,
            screenshot_width=width,
            screenshot_height=height,
            **extra,
        )
        raw_count = len(logical)
        logger.info(
            "Hierarchy stage count",
            extra={
                "component": "processing.drawer",
                "event": "hierarchy.stage.count",
                "stage": "raw",
                "count": raw_count,
            },
        )

        if action:
            logical = parser.filter_by_action(elements=logical, action=action)
            logger.info(
                "Hierarchy stage count",
                extra={
                    "component": "processing.drawer",
                    "event": "hierarchy.stage.count",
                    "stage": "after_action_filter",
                    "count": len(logical),
                    "action": str(action),
                },
            )

        action_filtered_count = len(logical)
        filtered = parser.filter_and_deduplicate(elements=logical, action=action)
        deduped_count = len(filtered)
        logger.info(
            "Hierarchy stage count",
            extra={
                "component": "processing.drawer",
                "event": "hierarchy.stage.count",
                "stage": "after_dedup",
                "count": deduped_count,
                "dropped_by_dedup": action_filtered_count - deduped_count,
            },
        )

        cv_added = 0
        if cv_enabled:
            visual_controls = VisualControlLabeler.detect(
                image=image,
                existing_elements=filtered,
                scale_factor=factor,
            )
            if visual_controls:
                cv_added = len(visual_controls)
                filtered = [*filtered, *visual_controls]

        mapping: Dict[str, Any] = {}
        labeled = []
        logger.info(
            "Hierarchy stage count",
            extra={
                "component": "processing.drawer",
                "event": "hierarchy.stage.count",
                "stage": "final",
                "count": len(filtered),
                "cv_added": cv_added,
                "raw": raw_count,
            },
        )

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
