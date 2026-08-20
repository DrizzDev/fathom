from __future__ import annotations

from logging import getLogger
from typing import Any, Dict, Final, List, Mapping, NamedTuple, Set, Tuple

from fathom.schemas.actions import Bounds
from fathom.schemas.observation import ElementSource, PerceivedElement, ScreenObservation

logger = getLogger(__name__)


class AppendedManifestEntry(NamedTuple):
    """
    One perception-sourced entry appended to the XML manifest.

    Carries everything an overlay renderer needs to draw a box on the
    LLM-facing annotated image without reparsing the bounds string:
    the assigned numeric label, the original :class:`ElementSource`,
    the optional element text (for label fallback), and the pixel
    bounds tuple already extracted from :class:`PerceivedElement`.
    """

    label_id: str
    source: ElementSource
    text: str
    bounds: Tuple[int, int, int, int]


class ManifestMergeResult(NamedTuple):
    """
    Outcome of :meth:`ManifestMerger.merge`.

    ``label_map`` is the enriched manifest the planner reads.
    ``appended`` lists only the perception entries newly added on top
    of the input XML+CV map — the overlay renderer iterates this
    list to draw matching boxes onto the LLM-facing annotated image.
    """

    label_map: Dict[str, Any]
    appended: Tuple[AppendedManifestEntry, ...]


class ManifestMerger:
    """
    Append text-bearing perception elements onto the XML-derived label
    map so the planner can bind ``label_id`` references against any
    visible text — regardless of which detector found it (OCR, icon
    detector, vision localizer).

    XML entries stay primary: numeric labels emitted by
    :class:`BoundsGenerator` are preserved untouched. Perception
    elements that carry semantic text and do not duplicate an existing
    XML region (by IoU threshold) are appended with continuation
    numeric labels in detection order. Bounds are serialised in the
    same ``[x1,y1][x2,y2]`` form the drawer uses so downstream
    consumers — manifest formatter, resolution snap — read both
    sources identically.
    """

    __PERCEPTION_SOURCES: Final[Set[ElementSource]] = {
        ElementSource.OCR,
        ElementSource.ICON,
        ElementSource.VISION,
    }
    __DUPLICATE_IOU_THRESHOLD: Final[float] = 0.5
    __SOURCE_KIND: Final[Mapping[ElementSource, str]] = {
        ElementSource.OCR: "OcrText",
        ElementSource.ICON: "IconMatch",
        ElementSource.VISION: "VisionRegion",
    }
    __SCALE_KEY: Final[str] = "__scale_factor__"

    @classmethod
    def merge(
        cls,
        *,
        label_map: Dict[str, Any],
        observation: ScreenObservation,
    ) -> ManifestMergeResult:
        """
        Return a :class:`ManifestMergeResult` with the enriched label map and the newly appended
        perception entries.

        The input ``label_map`` is not mutated; when no perception element qualifies (empty
        observation, no text, or all overlap the XML manifest), the result carries a shallow copy
        and an empty ``appended`` tuple. The appended entries are returned explicitly so the overlay
        renderer draws boxes without diffing maps or re-parsing the ``[x1,y1][x2,y2]`` bounds string.
        """

        enriched: Dict[str, Any] = dict(label_map)
        next_index = cls.__next_label_index(label_map=enriched)
        existing_bounds = cls.__existing_bounds(label_map=enriched)
        appended_entries: List[AppendedManifestEntry] = []

        for element in observation.elements:
            if not cls.__should_append(element=element):
                continue
            if cls.__overlaps_existing(bounds=element.bounds, existing_bounds=existing_bounds):
                continue

            label = str(next_index)
            enriched[label] = cls.__entry_from_element(element=element)
            x1 = int(element.bounds.x)
            y1 = int(element.bounds.y)
            x2 = int(element.bounds.x + element.bounds.width)
            y2 = int(element.bounds.y + element.bounds.height)
            existing_bounds.append((x1, y1, x2, y2))
            appended_entries.append(
                AppendedManifestEntry(
                    label_id=label,
                    source=element.source,
                    text=element.text or "",
                    bounds=(x1, y1, x2, y2),
                ),
            )
            next_index += 1

        if appended_entries:
            logger.info(
                "Merged perception elements into manifest",
                extra={
                    "component": "manifest.merger",
                    "event": "manifest.perception.merged",
                    "appended.count": len(appended_entries),
                    "xml.count": len(label_map) - (1 if cls.__SCALE_KEY in label_map else 0),
                },
            )

        return ManifestMergeResult(label_map=enriched, appended=tuple(appended_entries))

    @classmethod
    def __should_append(cls, *, element: PerceivedElement) -> bool:
        """
        Append filter: text-bearing perception sources only.
        """

        if element.source not in cls.__PERCEPTION_SOURCES:
            return False
        return bool(element.text and element.text.strip())

    @classmethod
    def __overlaps_existing(
        cls,
        *,
        bounds: Bounds,
        existing_bounds: list[Tuple[int, int, int, int]],
    ) -> bool:
        """
        Drop perception elements whose bounds substantially overlap a
        manifest entry already in the map — keeping one canonical
        anchor per region.
        """

        candidate = (
            bounds.x,
            bounds.y,
            bounds.x + bounds.width,
            bounds.y + bounds.height,
        )
        return any(
            cls.__iou(first=candidate, second=other) >= cls.__DUPLICATE_IOU_THRESHOLD
            for other in existing_bounds
        )

    @classmethod
    def __entry_from_element(cls, *, element: PerceivedElement) -> Dict[str, Any]:
        """
        Build a manifest entry from a perception element, mirroring the
        shape :class:`BoundsGenerator` emits so downstream consumers
        cannot tell the two apart.
        """

        x1 = int(element.bounds.x)
        y1 = int(element.bounds.y)
        x2 = int(element.bounds.x + element.bounds.width)
        y2 = int(element.bounds.y + element.bounds.height)
        kind = cls.__SOURCE_KIND.get(element.source, "PerceivedElement")
        return {
            "class": kind,
            "text": element.text or "",
            "bounds": f"[{x1},{y1}][{x2},{y2}]",
            "center_x": (x1 + x2) // 2,
            "center_y": (y1 + y2) // 2,
            "logical_bounds": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "source": element.source.value,
            "role": element.role.value,
            "confidence": element.confidence,
            "tappable": element.tappable,
        }

    @classmethod
    def __next_label_index(cls, *, label_map: Dict[str, Any]) -> int:
        """
        Highest existing numeric label + 1, or 1 when the map is empty.
        """

        highest = 0
        for label in label_map:
            if label.startswith("__"):
                continue
            try:
                highest = max(highest, int(label))
            except ValueError:
                continue
        return highest + 1

    @classmethod
    def __existing_bounds(cls, *, label_map: Dict[str, Any]) -> list[Tuple[int, int, int, int]]:
        """
        Project each entry's bounds string into a comparable rectangle
        for the IoU dedup pass.
        """

        rects: list[Tuple[int, int, int, int]] = []
        for label, info in label_map.items():
            if label.startswith("__") or not isinstance(info, dict):
                continue
            bounds_str = str(info.get("bounds", ""))
            if not bounds_str:
                continue
            parts = bounds_str.replace("][", ",").replace("[", "").replace("]", "").split(",")
            if len(parts) != 4:
                continue
            try:
                x1, y1, x2, y2 = (int(value) for value in parts)
            except ValueError:
                continue
            rects.append((x1, y1, x2, y2))
        return rects

    @staticmethod
    def __iou(
        *,
        first: Tuple[int, int, int, int],
        second: Tuple[int, int, int, int],
    ) -> float:
        """
        Intersection-over-Union of two ``(x1, y1, x2, y2)`` rectangles.
        """

        ax1, ay1, ax2, ay2 = first
        bx1, by1, bx2, by2 = second
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        iw = max(0, ix2 - ix1)
        ih = max(0, iy2 - iy1)
        intersection = iw * ih
        if intersection == 0:
            return 0.0
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - intersection
        if union <= 0:
            return 0.0
        return intersection / union
