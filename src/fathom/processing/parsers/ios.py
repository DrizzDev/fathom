from __future__ import annotations

from logging import getLogger
from math import hypot
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    import xml.etree.ElementTree as ET  # nosec

from fathom.constants import ActionType
from fathom.processing.geometry import GeometryUtils
from fathom.processing.parsers.base import PlatformParser
from fathom.schemas.ui import LabeledElement, UIBounds

logger = getLogger(__name__)


class IOSParser(PlatformParser):
    """
    XML parser for iOS Appium hierarchies.
    """

    __MIN_INTERACTIVE_DIMENSION = 5

    __MIN_HIDDEN_ICON_DIMENSION = 8
    __MAX_HIDDEN_ICON_DIMENSION = 40

    # iPhone point-based reference baseline for hidden-icon proximity scaling
    __HIDDEN_ICON_REFERENCE_WIDTH = 440
    __HIDDEN_ICON_REFERENCE_HEIGHT = 900

    __HIDDEN_ICON_ROW_Y_TOLERANCE = 40
    __HIDDEN_ICON_COLUMN_X_TOLERANCE = 40

    __HIDDEN_ICON_MAX_TEXT_GAP = 320
    __HIDDEN_ICON_MAX_VERTICAL_GAP = 320
    __HIDDEN_ICON_MAX_CENTER_DISTANCE = 240

    __TAPPABLE_TYPES = {
        "XCUIElementTypeKey",
        "XCUIElementTypeTab",
        "XCUIElementTypeIcon",
        "XCUIElementTypeCell",
        "XCUIElementTypeLink",
        "XCUIElementTypeImage",
        "XCUIElementTypeTabBar",
        "XCUIElementTypeButton",
        "XCUIElementTypeSwitch",
        "XCUIElementTypeStepper",
        "XCUIElementTypeToolbar",
        "XCUIElementTypeCheckBox",
        "XCUIElementTypeComboBox",
        "XCUIElementTypeRadioButton",
        "XCUIElementTypePopUpButton",
        "XCUIElementTypeIncrementArrow",
        "XCUIElementTypeDecrementArrow",
        "XCUIElementTypeSegmentedControl",
        "XCUIElementTypeDisclosureTriangle",
    }
    __TYPEABLE_TYPES = {
        "XCUIElementTypeComboBox",
        "XCUIElementTypeTextView",
        "XCUIElementTypeTextField",
        "XCUIElementTypeStaticText",
        "XCUIElementTypeSearchField",
        "XCUIElementTypeSecureTextField",
    }
    __SCROLLABLE_TYPES = {
        "XCUIElementTypeWebView",
        "XCUIElementTypeTextView",
        "XCUIElementTypeTableView",
        "XCUIElementTypeScrollView",
        "XCUIElementTypeCollectionView",
    }
    __DRAGGABLE_TYPES = {
        "XCUIElementTypeMap",
        "XCUIElementTypeSlider",
        "XCUIElementTypePicker",
        "XCUIElementTypePickerWheel",
        "XCUIElementTypeProgressIndicator",
    }
    __SWIPEABLE_TYPES = __SCROLLABLE_TYPES | {
        "XCUIElementTypeCell",
        "XCUIElementTypeTabBar",
        "XCUIElementTypePageIndicator",
    }
    __SYSTEM_ALERT_TYPES = {
        "XCUIElementTypeAlert",
        "XCUIElementTypeSheet",
        "XCUIElementTypeDialog",
    }
    __INTERACTIVE_TYPES = (
        __TAPPABLE_TYPES
        | __TYPEABLE_TYPES
        | __SWIPEABLE_TYPES
        | __DRAGGABLE_TYPES
        | __SYSTEM_ALERT_TYPES
        | {"XCUIElementTypeStaticText", "XCUIElementTypeImage"}
    )
    __CONTENT_TYPES = (
        __TAPPABLE_TYPES | __TYPEABLE_TYPES | {"XCUIElementTypeStaticText", "XCUIElementTypeImage"}
    )
    __GENERIC_CONTAINER_TYPES = {
        "XCUIElementTypeOther",
        "XCUIElementTypeWindow",
        "XCUIElementTypeWebView",
        "XCUIElementTypeApplication",
    }

    @classmethod
    def is_platform_match(cls, root: ET.Element) -> bool:
        """
        Return True when XML appears to be iOS hierarchy.
        """

        return any(element.get("type", "").startswith("XCUIElementType") for element in root.iter())

    def get_scale_factor(self, root: ET.Element, screenshot_width: int) -> float:
        """
        Calculate XML-to-image horizontal scale factor.
        """

        width, _ = self.__resolve_screen_dimensions(root=root)

        if width > 0:
            return screenshot_width / width

        return 1.0

    def find_all_elements(self, root: ET.Element, **extra: Any) -> List[LabeledElement]:
        """
        Extract interactive and meaningful elements from iOS XML.
        """

        detected: List[LabeledElement] = []

        screen_width, screen_height = self.__resolve_screen_dimensions(root=root, **extra)
        if screen_width == 0:
            logger.error("Could not determine iOS screen dimensions; skipping element extraction")
            return []

        hidden_icon_thresholds = self.__scaled_hidden_icon_thresholds(
            screen_width=screen_width, screen_height=screen_height
        )

        for node in root.iter():
            try:
                width = int(node.get("width", 0))
                height = int(node.get("height", 0))
                if not (width > 0 and height > 0):
                    continue

                x = int(node.get("x", 0))
                y = int(node.get("y", 0))

                if x + width <= 0 or x >= screen_width or y + height <= 0 or y >= screen_height:
                    continue

                metadata = {key: node.get(key, "") for key in node.attrib}

                hidden_icon_candidate = self.__is_hidden_tappable_icon_candidate(
                    x=x,
                    y=y,
                    root=root,
                    width=width,
                    height=height,
                    metadata=metadata,
                    max_text_gap=hidden_icon_thresholds["max_text_gap"],
                    row_y_tolerance=hidden_icon_thresholds["row_y_tolerance"],
                    max_vertical_gap=hidden_icon_thresholds["max_vertical_gap"],
                    column_x_tolerance=hidden_icon_thresholds["column_x_tolerance"],
                    max_center_distance=hidden_icon_thresholds["max_center_distance"],
                )

                if hidden_icon_candidate:
                    metadata["visible"] = "true"
                    metadata["hidden_tappable_icon_candidate"] = "true"

                if not self.__is_practically_interactive(
                    width=width,
                    height=height,
                    metadata=metadata,
                ):
                    continue

                if metadata.get("type") == "XCUIElementTypeApplication":
                    continue

                if (
                    "visible" in metadata
                    and str(metadata.get("visible", "")).lower() != "true"
                    and metadata.get("type") not in self.__INTERACTIVE_TYPES
                    and not hidden_icon_candidate
                ):
                    continue

                detected.append(
                    LabeledElement(
                        label="",
                        color="",
                        attributes=metadata,
                        bounds=UIBounds(x1=x, y1=y, x2=x + width, y2=y + height),
                    )
                )

            except (ValueError, TypeError) as exception:
                logger.warning(f"Failed to parse iOS element: {exception}")
                continue

        return detected

    @staticmethod
    def __clamp(value: int, minimum: int, maximum: int) -> int:
        """
        Clamp integer value into a closed interval.
        """

        return max(minimum, min(maximum, value))

    @classmethod
    def __scaled_hidden_icon_thresholds(
        cls, screen_width: int, screen_height: int
    ) -> Dict[str, int]:
        """
        Scale hidden-icon heuristics based on current screen size.
        """

        ref_width = max(1, cls.__HIDDEN_ICON_REFERENCE_WIDTH)
        ref_height = max(1, cls.__HIDDEN_ICON_REFERENCE_HEIGHT)

        scale_x = screen_width / ref_width
        scale_y = screen_height / ref_height

        scale_min = min(scale_x, scale_y)

        return {
            "row_y_tolerance": cls.__clamp(
                int(round(cls.__HIDDEN_ICON_ROW_Y_TOLERANCE * scale_y)), 24, 72
            ),
            "column_x_tolerance": cls.__clamp(
                int(round(cls.__HIDDEN_ICON_COLUMN_X_TOLERANCE * scale_x)), 24, 96
            ),
            "max_text_gap": cls.__clamp(
                int(round(cls.__HIDDEN_ICON_MAX_TEXT_GAP * scale_x)), 180, 560
            ),
            "max_vertical_gap": cls.__clamp(
                int(round(cls.__HIDDEN_ICON_MAX_VERTICAL_GAP * scale_y)), 180, 560
            ),
            "max_center_distance": cls.__clamp(
                int(round(cls.__HIDDEN_ICON_MAX_CENTER_DISTANCE * scale_min)), 120, 360
            ),
        }

    def __resolve_screen_dimensions(self, root: ET.Element, **extra: Any) -> Tuple[int, int]:
        """
        Resolve screen dimensions from root app node or fallback inputs.
        """

        app: Optional[ET.Element]

        if root.tag == "XCUIElementTypeApplication":
            app = root
        else:
            app = root.find(".//XCUIElementTypeApplication")

        if app is not None and int(app.get("width", "0")) > 0:
            return int(app.get("width", "0")), int(app.get("height", "0"))

        dimensions = next(
            (
                (int(node.get("width", "0")), int(node.get("height", "0")))
                for node in root.iter()
                if int(node.get("width", "0")) > 0
            ),
            (0, 0),
        )

        if dimensions[0] > 0:
            return dimensions

        return int(extra.get("screenshot_width", 0)), int(extra.get("screenshot_height", 0))

    def __is_practically_interactive(
        self, width: int, height: int, metadata: Dict[str, Any]
    ) -> bool:
        """
        Check whether an element is likely actionable in practice.
        """

        return not (
            (width < self.__MIN_INTERACTIVE_DIMENSION or height < self.__MIN_INTERACTIVE_DIMENSION)
            and not self.__has_semantic_content(metadata=metadata)
        )

    @staticmethod
    def __has_semantic_content(metadata: Dict[str, Any]) -> bool:
        """
        Return True when metadata contains user-meaningful text.
        """

        return bool(metadata.get("name") or metadata.get("label") or metadata.get("value"))

    @staticmethod
    def __interval_gap(start_a: int, end_a: int, start_b: int, end_b: int) -> int:
        """
        Compute distance between two 1D closed intervals.
        """

        if end_a < start_b:
            return start_b - end_a

        if end_b < start_a:
            return start_a - end_b

        return 0

    def __has_nearby_semantic_anchor(
        self,
        *,
        x: int,
        y: int,
        width: int,
        height: int,
        root: ET.Element,
        max_text_gap: int,
        row_y_tolerance: int,
        max_vertical_gap: int,
        column_x_tolerance: int,
        max_center_distance: int,
    ) -> bool:
        """
        Detect nearby semantic anchor around a hidden icon candidate.
        """

        candidate_center_x = x + (width / 2.0)
        candidate_center_y = y + (height / 2.0)

        candidate_x1, candidate_x2 = x, x + width
        candidate_y1, candidate_y2 = y, y + height

        for node in root.iter():
            try:
                node_width = int(node.get("width", 0))
                node_height = int(node.get("height", 0))

                if node_width <= 0 or node_height <= 0:
                    continue

                metadata = {key: node.get(key, "") for key in node.attrib}

                if str(metadata.get("visible", "")).lower() != "true":
                    continue

                if not self.__has_semantic_content(metadata=metadata):
                    continue

                node_x = int(node.get("x", 0))
                node_y = int(node.get("y", 0))
                text_x1, text_x2 = node_x, node_x + node_width
                text_y1, text_y2 = node_y, node_y + node_height

                text_center_x = node_x + (node_width / 2.0)
                text_center_y = node_y + (node_height / 2.0)

                horizontal_gap = self.__interval_gap(candidate_x1, candidate_x2, text_x1, text_x2)
                vertical_gap = self.__interval_gap(candidate_y1, candidate_y2, text_y1, text_y2)

                row_aligned = (
                    abs(text_center_y - candidate_center_y) <= row_y_tolerance
                    and horizontal_gap <= max_text_gap
                )
                if row_aligned:
                    return True

                column_aligned = (
                    abs(text_center_x - candidate_center_x) <= column_x_tolerance
                    and vertical_gap <= max_vertical_gap
                )
                if column_aligned:
                    return True

                center_distance = hypot(
                    text_center_x - candidate_center_x, text_center_y - candidate_center_y
                )
                if center_distance <= max_center_distance:
                    return True

            except (ValueError, TypeError):
                continue

        return False

    def __is_hidden_tappable_icon_candidate(
        self,
        *,
        x: int,
        y: int,
        width: int,
        height: int,
        root: ET.Element,
        max_text_gap: int,
        row_y_tolerance: int,
        max_vertical_gap: int,
        column_x_tolerance: int,
        metadata: Dict[str, Any],
        max_center_distance: int,
    ) -> bool:
        """
        Classify an element as hidden tappable icon candidate.
        """

        if metadata.get("type") != "XCUIElementTypeOther":
            return False

        if str(metadata.get("visible", "")).lower() == "true":
            return False

        if str(metadata.get("enabled", "")).lower() != "true":
            return False

        if str(metadata.get("accessible", "")).lower() == "true":
            return False

        if not (
            self.__MIN_HIDDEN_ICON_DIMENSION <= width <= self.__MAX_HIDDEN_ICON_DIMENSION
            and self.__MIN_HIDDEN_ICON_DIMENSION <= height <= self.__MAX_HIDDEN_ICON_DIMENSION
        ):
            return False

        if self.__has_semantic_content(metadata=metadata):
            return False

        return self.__has_nearby_semantic_anchor(
            x=x,
            y=y,
            root=root,
            width=width,
            height=height,
            max_text_gap=max_text_gap,
            row_y_tolerance=row_y_tolerance,
            max_vertical_gap=max_vertical_gap,
            column_x_tolerance=column_x_tolerance,
            max_center_distance=max_center_distance,
        )

    def __is_tappable(self, element: LabeledElement) -> bool:
        """
        Return True when element should be considered tappable.
        """

        metadata = element.attributes
        kind = metadata.get("type")

        if kind in self.__TAPPABLE_TYPES | self.__TYPEABLE_TYPES:
            return True

        if kind in self.__GENERIC_CONTAINER_TYPES:
            if str(metadata.get("hidden_tappable_icon_candidate", "")).lower() == "true":
                return True

            if "accessible" in metadata:
                return str(metadata.get("accessible", "")).lower() == "true"

            return True

        return False

    def __is_typeable(self, element: LabeledElement) -> bool:
        """
        Return True when element should accept text input.
        """

        return element.attributes.get("type") in self.__TYPEABLE_TYPES

    def __is_swipeable(self, element: LabeledElement) -> bool:
        """
        Return True when element should be treated as swipeable.
        """

        metadata = element.attributes
        kind = metadata.get("type")

        if kind in self.__SWIPEABLE_TYPES | self.__DRAGGABLE_TYPES:
            return True

        if kind in self.__GENERIC_CONTAINER_TYPES:
            if "accessible" in metadata:
                return str(metadata.get("accessible", "")).lower() == "true"

            return True

        return False

    def __score_element(self, element: LabeledElement, action: Any = None) -> float:
        """
        Score element utility for overlap suppression and pruning.
        """

        score = 0.0
        metadata = element.attributes

        if self.__is_swipeable(element=element):
            score += 150

        elif self.__is_tappable(element=element) or self.__is_typeable(element=element):
            score += 100

        if metadata.get("type") in self.__CONTENT_TYPES:
            score += 20

        if metadata.get("name") or metadata.get("label"):
            score += 10

        if metadata.get("type") in self.__GENERIC_CONTAINER_TYPES:
            score += 1

        if str(metadata.get("visible", "")).lower() == "true":
            score += 100

        if action == ActionType.TEXT or action == ActionType.TYPE:
            if self.__is_typeable(element=element):
                score += 100

        elif action == ActionType.TAP:
            if self.__is_tappable(element=element):
                score += 100

        elif action == ActionType.SWIPE and self.__is_swipeable(element=element):
            score += 100

        score -= element.bounds.area / 50000.0
        return score

    def __prune_containers(
        self, elements: List[LabeledElement], action: Any = None
    ) -> List[LabeledElement]:
        """
        Remove generic containers overshadowed by better child elements.
        """

        retained: List[LabeledElement] = []

        for index, parent in enumerate(elements):
            is_mere_container = False

            if parent.attributes.get("type") in self.__GENERIC_CONTAINER_TYPES:
                for next_index, child in enumerate(elements):
                    if index == next_index:
                        continue

                    if (
                        GeometryUtils.is_box_contained(box1=child.bounds, box2=parent.bounds)
                        and child.bounds.area < parent.bounds.area
                        and self.__score_element(element=child, action=action)
                        > self.__score_element(element=parent, action=action)
                    ):
                        is_mere_container = True
                        break

            if not is_mere_container:
                retained.append(parent)

        return retained

    def __suppress_overlaps_nms(
        self,
        elements: List[LabeledElement],
        *,
        iou_threshold: float = 0.4,
        action: Optional[Any] = None,
        score_diff_threshold: float = 50.0,
    ) -> List[LabeledElement]:
        """
        Apply non-maximum suppression while preserving strong contenders.
        """

        if not elements:
            return []

        kept: List[LabeledElement] = []
        scored = [
            (self.__score_element(element=element, action=action), element) for element in elements
        ]
        sorted_elements = sorted(scored, key=lambda item: item[0], reverse=True)

        while sorted_elements:
            best_score, best = sorted_elements.pop(0)
            kept.append(best)

            remaining = []
            for score, current in sorted_elements:
                iou = GeometryUtils.calculate_iou(bounds1=best.bounds, bounds2=current.bounds)

                if iou > iou_threshold:
                    same_bounds = (
                        best.bounds.x1 == current.bounds.x1
                        and best.bounds.y1 == current.bounds.y1
                        and best.bounds.x2 == current.bounds.x2
                        and best.bounds.y2 == current.bounds.y2
                    )

                    if same_bounds:
                        continue

                    if GeometryUtils.is_box_contained(box1=current.bounds, box2=best.bounds):
                        remaining.append((score, current))
                        continue

                    score_delta = abs(best_score - score)
                    if score_delta <= score_diff_threshold:
                        remaining.append((score, current))
                else:
                    remaining.append((score, current))

            sorted_elements = remaining

        return kept

    def __filter_for_meaningfulness(self, elements: List[LabeledElement]) -> List[LabeledElement]:
        """
        Retain elements with actionable or semantic relevance.
        """

        meaningful: List[LabeledElement] = []

        for element in elements:
            if (
                self.__is_tappable(element=element)
                or self.__is_typeable(element=element)
                or self.__is_swipeable(element=element)
                or element.attributes.get("name")
                or element.attributes.get("label")
            ):
                meaningful.append(element)

        return meaningful

    def filter_and_deduplicate(
        self,
        elements: List[LabeledElement],
        iou_threshold: float = 0.4,
        action: Optional[Any] = None,
    ) -> List[LabeledElement]:
        """
        Run full parser post-processing and sorting pipeline.
        """

        pruned = self.__prune_containers(elements=elements, action=action)
        suppressed = self.__suppress_overlaps_nms(
            action=action,
            elements=pruned,
            iou_threshold=iou_threshold,
        )
        meaningful = self.__filter_for_meaningfulness(elements=suppressed)

        return sorted(meaningful, key=lambda element: element.bounds.area, reverse=True)

    def filter_by_action(self, elements: List[LabeledElement], action: Any) -> List[LabeledElement]:
        """
        Filter parsed elements to action-compatible candidates.
        """

        if action == ActionType.TAP:
            return [element for element in elements if self.__is_tappable(element=element)]

        if action in {ActionType.TEXT, ActionType.TYPE}:
            return [
                element
                for element in elements
                if self.__is_typeable(element=element) or self.__is_tappable(element=element)
            ]

        if action == ActionType.SWIPE:
            return [element for element in elements if self.__is_swipeable(element=element)]

        return elements
