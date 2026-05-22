from __future__ import annotations

from logging import getLogger
from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:
    import xml.etree.ElementTree as ET  # nosec

from fathom.constants import ActionType
from fathom.constants.screen import REPEATED_TEXT_SUPPRESSION_THRESHOLD
from fathom.processing.geometry import GeometryUtils
from fathom.processing.parsers.base import PlatformParser
from fathom.schemas.hierarchy import NormalizedHierarchyNodeSignature
from fathom.schemas.ui import LabeledElement, UIBounds

logger = getLogger(__name__)


class AndroidParser(PlatformParser):
    """
    An advanced XML Parser for the Android platform.
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
    __SIGNATURE_IGNORE_IDENTIFIER_TOKENS = {
        "nav_bar",
        "systemui",
        "statusbar",
        "status_bar",
        "navigationbar",
    }

    @classmethod
    def is_platform_match(cls, root: ET.Element) -> bool:
        """
        Check if the platform matches.
        """

        return bool(root.get("package")) or root.find(".//*[@package]") is not None

    def get_scale_factor(self, root: ET.Element, screenshot_width: int) -> float:
        """
        Calculate the scale factor.
        """

        return 1.0

    def __is_practically_interactive(
        self, width: float, height: float, metadata: Dict[str, Any]
    ) -> bool:
        """
        Checks if an element is large enough to be interactive.
        """

        if width < self.__MIN_INTERACTIVE_DIMENSION or height < self.__MIN_INTERACTIVE_DIMENSION:
            has_id = bool(metadata.get("resource-id"))
            has_text = bool(str(metadata.get("text", "")).strip())
            has_description = bool(str(metadata.get("content-desc", "")).strip())

            if not has_text and not has_description and not has_id:
                return False

        return True

    def __is_meaningless_container(self, metadata: Dict[str, Any]) -> bool:
        """
        Checks if an element is a meaningless container.
        """

        kind = metadata.get("class", "")
        has_text = bool(str(metadata.get("text", "")).strip())
        is_enabled = str(metadata.get("enabled", "true")).lower() == "true"

        has_id = bool(metadata.get("resource-id", ""))
        has_description = bool(str(metadata.get("content-desc", "")).strip())

        is_focusable = str(metadata.get("focusable", "false")).lower() == "true"
        is_clickable = str(metadata.get("clickable", "false")).lower() == "true"
        is_scrollable = str(metadata.get("scrollable", "false")).lower() == "true"

        return (
            kind in self.__GENERIC_CONTAINER_CLASSES
            and not has_text
            and not is_enabled
            and not is_focusable
            and not is_clickable
            and not is_scrollable
            and not has_id
            and not has_description
        )

    def __is_tappable(self, element: LabeledElement) -> bool:
        """
        Checks if an element is tappable.
        """

        metadata = element.attributes
        kind = metadata.get("class")

        return (
            self.__is_typeable(element=element)
            or bool(metadata.get("resource-id"))
            or kind in self.__TAPPABLE_CLASSES
            or str(metadata.get("clickable")).lower() == "true"
            or bool(str(metadata.get("content-desc", "")).strip())
        )

    def __is_typeable(self, element: LabeledElement) -> bool:
        """
        Checks if an element is typeable.
        """

        metadata = element.attributes
        kind = metadata.get("class")

        if kind in self.__TYPEABLE_CLASSES:
            return True

        if kind in self.__GENERIC_CONTAINER_CLASSES and "enabled" in metadata:
            return str(metadata.get("enabled", "")).lower() == "true"

        return False

    def __is_swipeable(self, element: LabeledElement) -> bool:
        """
        Checks if an element is swipeable.
        """

        metadata = element.attributes
        kind = metadata.get("class")

        return (
            str(metadata.get("scrollable")).lower() == "true"
            or kind in self.__SWIPEABLE_CLASSES | self.__DRAGGABLE_CLASSES
        )

    def build_signature_metadata(self, *, node: ET.Element) -> NormalizedHierarchyNodeSignature:
        """
        Normalize one Android node into structural-signature metadata.
        """

        return NormalizedHierarchyNodeSignature(
            raw_value="",
            include_value_in_signature=False,
            bounds=str(node.get("bounds", "")),
            text=str(node.get("text", "")).strip(),
            source_type=str(node.get("class", node.tag)),
            class_name=str(node.get("class", node.tag)).split(".")[-1],
            identifier=str(node.get("resource-id", "")).split("/")[-1],
            content_description=str(node.get("content-desc", "")).strip(),
            is_focused=str(node.get("focused", "false")).lower() == "true",
            is_checked=str(node.get("checked", "false")).lower() == "true",
            is_selected=str(node.get("selected", "false")).lower() == "true",
            is_scrollable=str(node.get("scrollable", "false")).lower() == "true",
        )

    def should_ignore_signature_node(self, *, metadata: NormalizedHierarchyNodeSignature) -> bool:
        """
        Return whether an Android node should be excluded from the signature.
        """

        lowered_identifier = metadata.identifier.lower()

        return any(
            token in lowered_identifier for token in self.__SIGNATURE_IGNORE_IDENTIFIER_TOKENS
        )

    def find_all_elements(self, root: ET.Element, **extra: Any) -> List[LabeledElement]:
        """
        Finds all visible and interactive elements.
        """

        detected = []
        skip_displayed = extra.get("skip_displayed_attr", True)

        width = int(str(extra.get("screenshot_width", root.get("width", "0"))))
        height = int(str(extra.get("screenshot_height", root.get("height", "0"))))

        logger.info(f"Screen size: {width}x{height}")

        root_bounds = root.get("bounds")
        logger.info(f"Root XML bounds: {root_bounds}")

        skips = {
            "visibility": 0,
            "off_screen": 0,
            "meaningless": 0,
            "invalid_size": 0,
            "non_interactive": 0,
        }

        for node in root.iter():
            try:
                serialization = node.get("bounds")
                if not serialization:
                    continue

                parts = (
                    serialization.replace("][", ",").replace("[", "").replace("]", "").split(",")
                )

                if len(parts) != 4:
                    logger.warning(f"Invalid bounds format: {serialization}")
                    continue

                x1, y1, x2, y2 = map(int, parts)

                w, h = x2 - x1, y2 - y1
                metadata = {key: node.get(key, "") for key in node.attrib}
                metadata.update(
                    self.__scroll_metadata(
                        metadata=metadata,
                        width=x2 - x1,
                        height=y2 - y1,
                        screen_width=width,
                        screen_height=height,
                    )
                )

                if w <= 0 or h <= 0:
                    skips["invalid_size"] += 1
                    continue

                if not skip_displayed:
                    displayed = node.get("displayed")
                    visible = node.get("visible-to-user")

                    if (displayed and str(displayed).lower() == "false") or (
                        visible and str(visible).lower() == "false"
                    ):
                        skips["visibility"] += 1
                        continue

                if width > 0 and height > 0 and (x2 <= 0 or x1 >= width or y2 <= 0 or y1 >= height):
                    skips["off_screen"] += 1
                    continue

                if not self.__is_practically_interactive(
                    width=float(w), height=float(h), metadata=metadata
                ):
                    skips["non_interactive"] += 1
                    continue

                if self.__is_meaningless_container(metadata=metadata):
                    skips["meaningless"] += 1
                    continue

                detected.append(
                    LabeledElement(
                        label="",
                        color="",
                        attributes=metadata,
                        bounds=UIBounds(x1=x1, y1=y1, x2=x2, y2=y2),
                    )
                )

            except (ValueError, IndexError, TypeError) as exception:
                logger.warning(f"Failed to process element: {exception}")
                continue

        logger.info(f"Found {len(detected)} valid elements. Skips: {skips}")
        return detected

    def __scroll_metadata(
        self,
        *,
        metadata: Dict[str, Any],
        width: int,
        height: int,
        screen_width: int,
        screen_height: int,
    ) -> Dict[str, str]:
        """
        Attach normalized scroll metadata for downstream scope resolution.
        """

        kind = str(metadata.get("class", ""))
        explicitly_scrollable = str(metadata.get("scrollable", "false")).lower() == "true"
        scrollable = explicitly_scrollable or kind in self.__SCROLLABLE_CLASSES
        if not scrollable:
            return {}

        if kind == "android.widget.HorizontalScrollView":
            axis = "horizontal"
            scope_kind = "carousel"
        elif kind in {
            "androidx.viewpager.widget.ViewPager",
            "androidx.viewpager2.widget.ViewPager2",
        }:
            fills_viewport = width >= int(screen_width * 0.80) and height >= int(
                screen_height * 0.55
            )
            axis = "vertical" if fills_viewport else "horizontal"
            scope_kind = "viewport" if fills_viewport else "carousel"
        elif (
            explicitly_scrollable
            and width >= int(screen_width * 0.80)
            and height >= int(screen_height * 0.55)
        ):
            axis = "vertical"
            scope_kind = (
                "list"
                if kind
                in {
                    "android.widget.ListView",
                    "android.widget.GridView",
                    "android.widget.ScrollView",
                    "android.widget.ExpandableListView",
                    "androidx.core.widget.NestedScrollView",
                    "androidx.recyclerview.widget.RecyclerView",
                }
                else "viewport"
            )
        else:
            axis = "vertical"
            scope_kind = "list"
        return {
            "scrollable": "true",
            "axis": axis,
            "kind": scope_kind,
        }

    def filter_and_deduplicate(
        self,
        elements: List[LabeledElement],
        *,
        action: Any = None,
        iou_threshold: float = 0.4,
    ) -> List[LabeledElement]:
        """
        Orchestrates the filtering pipeline.
        """

        _ = action

        pruned = self.__prune_containers(elements=elements)
        suppressed = self.__suppress_overlaps(elements=pruned, threshold=iou_threshold)

        meaningful = self.__filter_for_meaningfulness(elements=suppressed)
        deduped = self.__suppress_repeated_decorative_text(elements=meaningful)
        return sorted(deduped, key=lambda element: element.bounds.area, reverse=True)

    @staticmethod
    def __suppress_repeated_decorative_text(
        elements: List[LabeledElement],
    ) -> List[LabeledElement]:
        """
        Collapse identical decorative-text labels that repeat across cards.

        Mirrors the iOS suppression: keeps the first occurrence of any
        repeated ``android.widget.TextView`` ``text`` so the planner
        does not pick the same label twice within the same screen.
        """

        seen: Dict[str, int] = {}
        retained: List[LabeledElement] = []
        for element in elements:
            kind = str(element.attributes.get("class") or "")
            if "TextView" not in kind:
                retained.append(element)
                continue
            key = str(element.attributes.get("text") or "").strip().lower()
            if not key:
                retained.append(element)
                continue
            seen[key] = seen.get(key, 0) + 1
            if seen[key] <= REPEATED_TEXT_SUPPRESSION_THRESHOLD - 1:
                retained.append(element)
        return retained

    def filter_by_action(self, elements: List[LabeledElement], action: Any) -> List[LabeledElement]:
        """
        Filters elements relevant to a specific action.
        """

        if action == ActionType.TAP:
            return [element for element in elements if self.__is_tappable(element=element)]

        if action == ActionType.TEXT or action == ActionType.TYPE:
            return [
                element
                for element in elements
                if self.__is_typeable(element=element) or self.__is_tappable(element=element)
            ]

        if action == ActionType.SWIPE:
            return [element for element in elements if self.__is_swipeable(element=element)]

        return elements

    def __score_element(self, element: LabeledElement) -> float:
        """
        Scores an element.
        """

        score = 0.0
        metadata = element.attributes
        kind = metadata.get("class")

        if self.__is_swipeable(element=element):
            score += 150

        elif self.__is_tappable(element=element):
            score += 100

        if kind in self.__CONTENT_CLASSES:
            score += 20

        if metadata.get("text") or metadata.get("content-desc"):
            score += 10

        if kind in self.__GENERIC_CONTAINER_CLASSES:
            score += 1

        score -= element.bounds.area / 50000.0
        return score

    def __prune_containers(self, elements: List[LabeledElement]) -> List[LabeledElement]:
        """
        Prunes containers that are mere wrappers.
        """

        retained = []

        for index, parent in enumerate(iterable=elements):
            is_mere_container = False

            is_candidate = (
                not str(parent.attributes.get("text", "")).strip()
                and not str(parent.attributes.get("content-desc", "")).strip()
                and str(parent.attributes.get("clickable", "false")).lower() == "false"
                and str(parent.attributes.get("scrollable", "false")).lower() == "false"
                and parent.attributes.get("class") in self.__GENERIC_CONTAINER_CLASSES
            )

            if is_candidate:
                for next_index, child in enumerate(iterable=elements):
                    if index == next_index:
                        continue

                    if (
                        parent.bounds.x1 <= child.bounds.x1
                        and parent.bounds.y1 <= child.bounds.y1
                        and parent.bounds.x2 >= child.bounds.x2
                        and parent.bounds.y2 >= child.bounds.y2
                        and self.__score_element(element=child)
                        > self.__score_element(element=parent)
                    ):
                        is_mere_container = True
                        break

            if not is_mere_container:
                retained.append(parent)

        return retained

    def __suppress_overlaps(
        self, elements: List[LabeledElement], threshold: float = 0.4
    ) -> List[LabeledElement]:
        """
        Non-Maximum Suppression for overlapping elements.
        """

        if not elements:
            return []

        kept = []
        scored = [(self.__score_element(element=element), element) for element in elements]
        sorted_elements = sorted(scored, key=lambda item: item[0], reverse=True)

        while sorted_elements:
            _, best = sorted_elements.pop(0)
            kept.append(best)

            remaining = []
            for score, current in sorted_elements:
                intersection = GeometryUtils.calculate_iou(
                    bounds1=best.bounds, bounds2=current.bounds
                )

                if intersection > threshold:
                    match = (
                        best.bounds.x1 == current.bounds.x1
                        and best.bounds.y1 == current.bounds.y1
                        and best.bounds.x2 == current.bounds.x2
                        and best.bounds.y2 == current.bounds.y2
                    )

                    if match:
                        pass
                    elif GeometryUtils.is_box_contained(
                        box1=current.bounds, box2=best.bounds
                    ) or GeometryUtils.is_box_contained(box1=best.bounds, box2=current.bounds):
                        remaining.append((score, current))
                    else:
                        pass
                else:
                    remaining.append((score, current))

            sorted_elements = remaining

        return kept

    def __filter_for_meaningfulness(self, elements: List[LabeledElement]) -> List[LabeledElement]:
        """
        Removes decorative or structural elements.
        """
        meaningful = []
        for element in elements:
            metadata = element.attributes
            if (
                self.__is_tappable(element=element)
                or self.__is_swipeable(element=element)
                or metadata.get("class") in self.__CONTENT_CLASSES
                or (
                    str(metadata.get("text", "")).strip()
                    or str(metadata.get("content-desc", "")).strip()
                )
            ):
                meaningful.append(element)

        return meaningful
