from __future__ import annotations

from logging import getLogger
from typing import Any, Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    import xml.etree.ElementTree as ET  # nosec

from fathom.constants import ActionType
from fathom.schemas.ui import LabeledElement, UIBoundingBox
from fathom.tools.vision.processing.geometry import GeometryUtils
from fathom.tools.vision.processing.parsers.base import PlatformParser

logger = getLogger(__name__)


class AndroidParser(PlatformParser):
    """
    An advanced Appium XML Parser for the Android platform that uses a multi-stage
    filtering pipeline to accurately identify all meaningful visible elements.
    """

    __MIN_INTERACTIVE_DIMENSION = 5

    __TAPPABLE_CLASSES = {
        "android.widget.Button",
        "android.widget.Switch",
        "android.widget.Spinner",
        "android.widget.SeekBar",
        "android.widget.TextView",
        "android.widget.CheckBox",
        "android.widget.RatingBar",
        "android.widget.ImageView",
        "android.widget.ImageButton",
        "android.widget.RadioButton",
        "android.widget.ToggleButton",
        "android.widget.CompoundButton",
    }

    __TYPEABLE_CLASSES = {
        "android.widget.EditText",
        "android.widget.AutoCompleteTextView",
        "android.widget.MultiAutoCompleteTextView",
    }

    __SCROLLABLE_CLASSES = {
        "android.widget.ListView",
        "android.widget.GridView",
        "android.widget.ScrollView",
        "android.widget.ExpandableListView",
        "android.widget.HorizontalScrollView",
        "androidx.viewpager.widget.ViewPager",
        "androidx.viewpager2.widget.ViewPager2",
        "androidx.core.widget.NestedScrollView",
        "androidx.recyclerview.widget.RecyclerView",
    }

    __SWIPEABLE_CLASSES = __SCROLLABLE_CLASSES | {
        "android.widget.TabHost",
        "android.widget.Gallery",
        "android.widget.ViewFlipper",
        "android.widget.ViewSwitcher",
    }

    __DRAGGABLE_CLASSES = {
        "android.widget.Switch",
        "android.widget.SeekBar",
        "android.widget.RatingBar",
        "android.widget.ProgressBar",
    }

    __CONTENT_CLASSES = (
        __TAPPABLE_CLASSES
        | __TYPEABLE_CLASSES
        | {
            "android.webkit.WebView",
            "android.widget.TextView",
            "android.widget.ImageView",
            "android.widget.VideoView",
        }
    )

    __GENERIC_CONTAINER_CLASSES = {
        "android.view.View",
        "android.view.ViewGroup",
        "android.widget.TableRow",
        "android.widget.GridLayout",
        "android.widget.TableLayout",
        "android.widget.FrameLayout",
        "android.widget.LinearLayout",
        "android.widget.RelativeLayout",
        "android.widget.ConstraintLayout",
        "androidx.appcompat.widget.LinearLayoutCompat",
        "androidx.constraintlayout.widget.ConstraintLayout",
        "androidx.coordinatorlayout.widget.CoordinatorLayout",
    }

    __SPECIALIZED_CONTAINERS = {
        "android.widget.Toolbar",
        "android.widget.ActionBar",
        "android.widget.TabWidget",
        "androidx.appcompat.app.ActionBar",
        "androidx.appcompat.widget.Toolbar",
        "com.google.android.material.tabs.TabLayout",
    }

    __INTERACTIVE_CLASSES = (
        __TAPPABLE_CLASSES | __TYPEABLE_CLASSES | __SCROLLABLE_CLASSES | __DRAGGABLE_CLASSES
    )

    @classmethod
    def is_platform_match(cls, root: ET.Element) -> bool:
        """
        Check if the platform matches.
        """
        return bool(root.get("package")) or root.find(".//*[@package]") is not None

    def get_scale_factor(self, root: ET.Element, screenshot_width: int) -> float:
        """
        Calculate the scale factor for the platform.
        """
        return 1.0

    def __is_practically_interactive(
        self, width: float, height: float, attributes: Dict[str, Any]
    ) -> bool:
        """
        Checks if an element is large enough to be interactive.
        """
        if (
            width < self.__MIN_INTERACTIVE_DIMENSION
            or height < self.__MIN_INTERACTIVE_DIMENSION
        ):
            has_text_content = bool(str(attributes.get("text", "")).strip())
            has_description_content = bool(str(attributes.get("content-desc", "")).strip())

            if not has_text_content and not has_description_content:
                return False

        return True

    def __is_meaningless_container(self, attributes: Dict[str, Any]) -> bool:
        """
        Checks if an element is a meaningless container.
        """
        element_class_name = attributes.get("class", "")
        has_text_content = bool(str(attributes.get("text", "")).strip())
        is_enabled_flag = str(attributes.get("enabled", "true")).lower() == "true"

        has_resource_id_flag = bool(attributes.get("resource-id", ""))
        has_content_description_flag = bool(str(attributes.get("content-desc", "")).strip())

        is_focusable_flag = str(attributes.get("focusable", "false")).lower() == "true"
        is_clickable_flag = str(attributes.get("clickable", "false")).lower() == "true"
        is_scrollable_flag = str(attributes.get("scrollable", "false")).lower() == "true"

        return (
            element_class_name in self.__GENERIC_CONTAINER_CLASSES
            and not has_text_content
            and not is_enabled_flag
            and not is_focusable_flag
            and not is_clickable_flag
            and not is_scrollable_flag
            and not has_resource_id_flag
            and not has_content_description_flag
        )

    def __is_tappable(self, element: LabeledElement) -> bool:
        """
        Checks if an element is tappable based on its attributes or class.
        """
        attributes = element.attributes
        class_name = attributes.get("class")

        return (
            str(attributes.get("clickable")).lower() == "true"
            or class_name in self.__TAPPABLE_CLASSES
            or self.__is_typeable(element)
        )

    def __is_typeable(self, element: LabeledElement) -> bool:
        """
        Checks if an element is typeable based on its class.
        """
        attributes = element.attributes
        class_name = attributes.get("class")

        if class_name in self.__TYPEABLE_CLASSES:
            return True

        if class_name in self.__GENERIC_CONTAINER_CLASSES and "enabled" in attributes:
            return str(attributes.get("enabled", "")).lower() == "true"

        return False

    def __is_swipeable(self, element: LabeledElement) -> bool:
        """
        Checks if an element is swipeable based on its attributes or class.
        """
        attributes = element.attributes
        class_name = attributes.get("class")

        return (
            str(attributes.get("scrollable")).lower() == "true"
            or class_name in self.__SWIPEABLE_CLASSES | self.__DRAGGABLE_CLASSES
        )

    def find_all_elements(
        self, root: ET.Element, **kwargs: Any
    ) -> List[LabeledElement]:
        """
        STAGE 1: Find all elements with valid size, position, and visibility.
        """
        should_skip_displayed_check = kwargs.get("skip_displayed_attr", True)
        detected_elements = []
        screen_width = int(str(kwargs.get("screenshot_width", root.get("width", "0"))))
        screen_height = int(
            str(kwargs.get("screenshot_height", root.get("height", "0")))
        )

        logger.info(f"Screen size: {screen_width}x{screen_height}")

        skip_reason_counts = {
            "visibility": 0,
            "off_screen": 0,
            "meaningless": 0,
            "invalid_size": 0,
            "non_interactive": 0,
        }

        for element in root.iter():
            try:
                bounds_string = element.get("bounds")
                if not bounds_string:
                    continue

                bounds_parts = (
                    bounds_string.replace("][", ",")
                    .replace("[", "")
                    .replace("]", "")
                    .split(",")
                )

                if len(bounds_parts) != 4:
                    logger.warning(f"Invalid bounds format: {bounds_string}")
                    continue

                x1, y1, x2, y2 = map(int, bounds_parts)

                width, height = x2 - x1, y2 - y1
                attributes = {key: element.get(key, "") for key in element.attrib}

                if width <= 0 or height <= 0:
                    skip_reason_counts["invalid_size"] += 1
                    continue

                if not should_skip_displayed_check:
                    displayed_status = element.get("displayed")
                    visible_status = element.get("visible-to-user")

                    if (displayed_status and str(displayed_status).lower() == "false") or (
                        visible_status and str(visible_status).lower() == "false"
                    ):
                        skip_reason_counts["visibility"] += 1
                        continue

                if screen_width > 0 and screen_height > 0 and (
                    x2 <= 0 or x1 >= screen_width or y2 <= 0 or y1 >= screen_height
                ):
                    skip_reason_counts["off_screen"] += 1
                    continue

                if not self.__is_practically_interactive(
                    width=float(width), height=float(height), attributes=attributes
                ):
                    skip_reason_counts["non_interactive"] += 1
                    continue

                if self.__is_meaningless_container(attributes=attributes):
                    skip_reason_counts["meaningless"] += 1
                    continue

                detected_elements.append(
                    LabeledElement(
                        label="",
                        color="",
                        attributes=attributes,
                        bounds=UIBoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                    )
                )

            except (ValueError, IndexError, TypeError) as exception:
                logger.warning(f"Failed to process element: {exception}")
                continue

        logger.info(f"Found {len(detected_elements)} valid elements. Skips: {skip_reason_counts}")
        return detected_elements

    def filter_and_deduplicate(
        self,
        elements: List[LabeledElement],
        iou_threshold: float = 0.4,
        action: Any = None,
    ) -> List[LabeledElement]:
        """
        Orchestrates the complete, corrected filtering pipeline.
        """
        pruned_elements = self.__prune_containers(elements)
        suppressed_elements = self.__suppress_overlaps_nms(
            pruned_elements, iou_threshold=iou_threshold
        )

        meaningful_elements = self.__filter_for_meaningfulness(suppressed_elements)
        return sorted(
            meaningful_elements, key=lambda element: element.bounds.area, reverse=True
        )

    def filter_by_action(
        self, elements: List[LabeledElement], action: Any
    ) -> List[LabeledElement]:
        """
        Filters elements relevant to a specific action.
        """
        if action == ActionType.TAP:
            return [element for element in elements if self.__is_tappable(element)]

        if action == ActionType.TEXT or action == ActionType.TYPE:
            return [
                element
                for element in elements
                if self.__is_typeable(element) or self.__is_tappable(element)
            ]

        if action == ActionType.SWIPE:
            return [element for element in elements if self.__is_swipeable(element)]

        return elements

    def __score_element(self, element: LabeledElement) -> float:
        """
        Scores an element based on its class, attributes, and size.
        """
        score_value = 0.0
        attributes = element.attributes
        class_name = attributes.get("class")

        if self.__is_swipeable(element):
            score_value += 150
        elif self.__is_tappable(element):
            score_value += 100

        if class_name in self.__CONTENT_CLASSES:
            score_value += 20

        if attributes.get("text") or attributes.get("content-desc"):
            score_value += 10

        if class_name in self.__GENERIC_CONTAINER_CLASSES:
            score_value += 1

        score_value -= element.bounds.area / 50000.0
        return score_value

    def __prune_containers(
        self, elements: List[LabeledElement]
    ) -> List[LabeledElement]:
        """
        Prunes containers that are mere wrappers for other elements.
        """
        elements_to_retain = []

        for index, parent in enumerate(elements):
            is_mere_container_flag = False

            is_candidate_for_pruning_flag = (
                not str(parent.attributes.get("text", "")).strip()
                and not str(parent.attributes.get("content-desc", "")).strip()
                and str(parent.attributes.get("clickable", "false")).lower() == "false"
                and str(parent.attributes.get("scrollable", "false")).lower() == "false"
                and parent.attributes.get("class") in self.__GENERIC_CONTAINER_CLASSES
            )

            if is_candidate_for_pruning_flag:
                for next_index, child in enumerate(elements):
                    if index == next_index:
                        continue

                    if (
                        parent.bounds.x1 <= child.bounds.x1
                        and parent.bounds.y1 <= child.bounds.y1
                        and parent.bounds.x2 >= child.bounds.x2
                        and parent.bounds.y2 >= child.bounds.y2
                        and self.__score_element(child) > self.__score_element(parent)
                    ):
                        is_mere_container_flag = True
                        break

            if not is_mere_container_flag:
                elements_to_retain.append(parent)

        return elements_to_retain

    def __suppress_overlaps_nms(
        self, elements: List[LabeledElement], iou_threshold: float = 0.4
    ) -> List[LabeledElement]:
        """
        STAGE 3: Uses a more nuanced Non-Maximum Suppression.
        """
        if not elements:
            return []

        kept_elements_list = []
        scored_elements_list = [
            (self.__score_element(element), element) for element in elements
        ]
        sorted_elements_list = sorted(
            scored_elements_list, key=lambda item: item[0], reverse=True
        )

        while sorted_elements_list:
            _, best_element = sorted_elements_list.pop(0)
            kept_elements_list.append(best_element)

            remaining_scored_elements = []
            for score, current_element in sorted_elements_list:
                intersection_over_union = GeometryUtils.calculate_iou(
                    best_element.bounds, current_element.bounds
                )

                if intersection_over_union > iou_threshold:
                    bounds_match = (
                        best_element.bounds.x1 == current_element.bounds.x1
                        and best_element.bounds.y1 == current_element.bounds.y1
                        and best_element.bounds.x2 == current_element.bounds.x2
                        and best_element.bounds.y2 == current_element.bounds.y2
                    )

                    if bounds_match:
                        pass
                    elif GeometryUtils.is_box_contained(
                        current_element.bounds, best_element.bounds
                    ):
                        remaining_scored_elements.append((score, current_element))
                    elif GeometryUtils.is_box_contained(
                        best_element.bounds, current_element.bounds
                    ):
                        remaining_scored_elements.append((score, current_element))
                    else:
                        pass
                else:
                    remaining_scored_elements.append((score, current_element))

            sorted_elements_list = remaining_scored_elements

        return kept_elements_list

    def __filter_for_meaningfulness(
        self, elements: List[LabeledElement]
    ) -> List[LabeledElement]:
        """
        Final pipeline stage to remove purely decorative or structural elements.
        """
        meaningful_elements_list = []
        for element in elements:
            attributes = element.attributes
            if (
                self.__is_tappable(element)
                or self.__is_swipeable(element)
                or attributes.get("class") in self.__CONTENT_CLASSES
                or (
                    str(attributes.get("text", "")).strip()
                    or str(attributes.get("content-desc", "")).strip()
                )
            ):
                meaningful_elements_list.append(element)
        return meaningful_elements_list