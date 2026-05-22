from __future__ import annotations

import asyncio
import time
from logging import getLogger
from typing import Any, Dict, List, Optional, Tuple

import numpy

try:
    import cv2
except ModuleNotFoundError:  # pragma: no cover - dependency optional when CV is disabled
    cv2 = None

from fathom.adapters.icon.noop import NoopIconDetector
from fathom.adapters.ocr.noop import NoopOcr
from fathom.adapters.perception.overlay.noop import NoopOverlayDetector
from fathom.constants.command import CommandScopeKind
from fathom.constants.perception import (
    BUTTON_CLASS_HINTS,
    CALL_TO_ACTION_MINIMUM_AREA,
    CALL_TO_ACTION_TEXT,
    INPUT_CLASS_HINTS,
    KEYBOARD_BOTTOM_REGION_RATIO,
    KEYBOARD_CLASS_HINTS,
    KEYBOARD_EDGE_DENSITY_THRESHOLD,
    KEYBOARD_MINIMUM_CONTOUR_COUNT,
    OCR_MAXIMUM_TOKEN_LENGTH,
    OCR_TRIGGER_MANIFEST_TEXT_COVERAGE,
    OVERLAY_MINIMUM_COVERAGE_RATIO,
    SCROLL_CLASS_HINTS,
    VISUAL_CONTROL_CONFIDENCE,
    VISUAL_CONTROL_MAXIMUM_HEIGHT_RATIO,
    VISUAL_CONTROL_MAXIMUM_WIDTH_RATIO,
    VISUAL_CONTROL_MINIMUM_AREA,
    VISUAL_CONTROL_MINIMUM_FILL_RATIO,
    VISUAL_CONTROL_MINIMUM_HEIGHT,
    VISUAL_CONTROL_MINIMUM_IOU,
    VISUAL_CONTROL_MINIMUM_SATURATION,
    VISUAL_CONTROL_MINIMUM_VALUE,
    VISUAL_CONTROL_MINIMUM_WIDTH,
)
from fathom.constants.scroll import ScrollEvidenceSource
from fathom.core.artifact.pipeline import ArtifactPipeline
from fathom.core.exceptions import OcrError
from fathom.interfaces.icon import IconDetectorPort
from fathom.interfaces.ocr import OcrPort
from fathom.interfaces.overlay import OverlayDetectorPort
from fathom.schemas.actions import Bounds, CoordinateSource, CoordinateSystem
from fathom.schemas.artifact import (
    ArtifactRecord,
    CvPerceptionPayload,
    IconPerceptionPayload,
    OcrPerceptionPayload,
    OverlayPerceptionPayload,
    PerceptionPayload,
    VisionPerceptionPayload,
)
from fathom.schemas.budgets import PerceptionBudget
from fathom.schemas.icon import IconMatch
from fathom.schemas.observation import (
    ElementRole,
    ElementSource,
    KeyboardObservation,
    OverlayObservation,
    PerceivedElement,
    ScreenObservation,
    ScrollRegion,
)
from fathom.schemas.ocr import OcrToken
from fathom.schemas.perception import PerceptionConfiguration
from fathom.schemas.screens import ScreenCapture, ScreenHashBundle
from fathom.schemas.ui import LabeledElement, UIBounds

logger = getLogger(__name__)


class ScreenObservationService:
    """
    Builds unified screen observations from capture and manifest inputs.
    """

    def __init__(
        self,
        *,
        workflow_id: Optional[str] = None,
        ocr: Optional[OcrPort] = None,
        icons: Optional[IconDetectorPort] = None,
        pipeline: Optional[ArtifactPipeline] = None,
        pixel_overlay: Optional[OverlayDetectorPort] = None,
        configuration: Optional[PerceptionConfiguration] = None,
    ) -> None:
        """
        Initialize the observation service with optional providers and run context.
        """

        self.__configuration = configuration or PerceptionConfiguration()
        self.__ocr = ocr if ocr is not None else NoopOcr()
        self.__icons = icons if icons is not None else NoopIconDetector()
        self.__pixel_overlay = pixel_overlay if pixel_overlay is not None else NoopOverlayDetector()
        self.__pipeline = pipeline
        self.__workflow_id = workflow_id

    async def observe(
        self,
        *,
        capture: ScreenCapture,
        hashes: ScreenHashBundle,
        budget: PerceptionBudget,
        manifest: Tuple[LabeledElement, ...],
        session_id: str,
        step_number: int,
    ) -> ScreenObservation:
        """
        Build a normalized screen observation.
        """

        elements = tuple(
            self.__element_from_label(element=element, index=index)
            for index, element in enumerate(manifest, start=1)
        )

        if self.__configuration.cv.enabled:
            if cv2 is None:
                raise RuntimeError("OpenCV is required when perception.cv.enabled is true.")
            visual_elements = self.__visual_controls(
                capture=capture,
                existing=elements,
                start=len(elements) + 1,
            )
            elements = (*elements, *visual_elements)
        elements = await self.__merge_async_enrichment(
            capture=capture,
            budget=budget,
            elements=elements,
        )

        keyboard = self.__keyboard(elements=elements, capture=capture)
        overlays = self.__overlays(elements=elements, capture=capture)
        if (
            pixel_overlay := await self.__pixel_overlay_observation(
                capture=capture,
                budget=budget,
                existing=overlays,
                elements=elements,
            )
        ) is not None:
            overlays = (*overlays, pixel_overlay)

        scroll = self.__scroll_regions(elements=elements, capture=capture)
        calls_to_action = self.__calls_to_action(elements=elements)

        observation = ScreenObservation(
            hashes=hashes,
            scroll=scroll,
            elements=elements,
            overlays=overlays,
            keyboard=keyboard,
            activity=capture.activity,
            calls_to_action=calls_to_action,
            focused=self.__focused(elements=elements),
        )
        await self.__emit_perception_artifact(
            capture=capture,
            observation=observation,
            session_id=session_id,
            step_number=step_number,
        )
        return observation

    async def __merge_async_enrichment(
        self,
        *,
        capture: ScreenCapture,
        budget: PerceptionBudget,
        elements: Tuple[PerceivedElement, ...],
    ) -> Tuple[PerceivedElement, ...]:
        """
        Merge optional OCR and icon enrichers without serializing their wall time.
        """

        base = elements
        ocr_task = None
        if self.__manifest_text_coverage(elements=base) < OCR_TRIGGER_MANIFEST_TEXT_COVERAGE:
            ocr_task = asyncio.create_task(
                self.__ocr_elements(
                    capture=capture,
                    budget=budget,
                    existing=base,
                    start=len(base) + 1,
                )
            )

        icon_task = asyncio.create_task(
            self.__icon_elements(
                capture=capture,
                budget=budget,
                existing=base,
                start=len(base) + 1,
            )
        )

        if ocr_task is None:
            icon_elements = await icon_task
            return (*base, *icon_elements)

        ocr_elements, icon_elements = await asyncio.gather(ocr_task, icon_task)
        return (*base, *ocr_elements, *icon_elements)

    async def __emit_perception_artifact(
        self,
        *,
        capture: ScreenCapture,
        observation: ScreenObservation,
        session_id: str,
        step_number: int,
    ) -> None:
        """
        Hand perception evidence to the artifact pipeline, but only the
        artifacts whose source actually contributed elements.

        Saving a merged perception image when observation contains only
        manifest/accessibility elements produces a second copy of the
        annotated hierarchy with no additional signal. We emit the
        merged artifact only when a true enrichment source contributed.
        """

        if self.__pipeline is None:
            return

        created = int(time.time() * 1000)
        sources = {element.source for element in observation.elements}
        enrichment_sources = {
            ElementSource.OCR,
            ElementSource.CV,
            ElementSource.ICON,
            ElementSource.VISION,
        }

        if any(source in enrichment_sources for source in sources):
            await self.__pipeline.emit(
                record=ArtifactRecord(
                    session_id=session_id,
                    package_name=capture.activity,
                    step_number=step_number,
                    created=created,
                    payload=PerceptionPayload(capture=capture, observation=observation),
                ),
            )
        for source, payload_factory in (
            (ElementSource.OCR, OcrPerceptionPayload),
            (ElementSource.CV, CvPerceptionPayload),
            (ElementSource.ICON, IconPerceptionPayload),
            (ElementSource.VISION, VisionPerceptionPayload),
        ):
            if source not in sources:
                continue
            await self.__pipeline.emit(
                record=ArtifactRecord(
                    session_id=session_id,
                    package_name=capture.activity,
                    step_number=step_number,
                    created=created,
                    payload=payload_factory(capture=capture, observation=observation),
                ),
            )
        if observation.overlays:
            await self.__pipeline.emit(
                record=ArtifactRecord(
                    session_id=session_id,
                    package_name=capture.activity,
                    step_number=step_number,
                    created=created,
                    payload=OverlayPerceptionPayload(
                        capture=capture,
                        observation=observation,
                    ),
                ),
            )

    async def __ocr_elements(
        self,
        *,
        capture: ScreenCapture,
        budget: PerceptionBudget,
        existing: Tuple[PerceivedElement, ...],
        start: int,
    ) -> Tuple[PerceivedElement, ...]:
        """
        Call the OCR port and convert returned tokens into perceived elements.
        """

        try:
            result = await self.__ocr.extract(capture=capture, budget=budget)
        except OcrError as exception:
            logger.warning(
                "OCR enrichment skipped",
                extra={
                    **self.__log_context(activity=capture.activity),
                    "event": "observation.ocr.skipped",
                    "retryable": exception.retryable,
                    "reason": exception.message,
                },
            )
            return ()

        tokens = tuple(
            token for token in result.tokens if len(token.text) <= OCR_MAXIMUM_TOKEN_LENGTH
        )
        if not tokens:
            logger.info(
                "OCR returned no usable tokens",
                extra={
                    **self.__log_context(activity=capture.activity),
                    "event": "observation.ocr.empty",
                    "raw.token.count": len(result.tokens),
                },
            )
            return ()

        merged: List[PerceivedElement] = []
        for offset, token in enumerate(tokens):
            if self.__overlaps_existing(bounds=token.bounds, existing=existing):
                continue
            merged.append(self.__element_from_token(token=token, index=start + offset))
        logger.info(
            "OCR tokens merged into observation",
            extra={
                **self.__log_context(activity=capture.activity),
                "event": "observation.ocr.merged",
                "merged.count": len(merged),
                "duration.ms": result.duration,
            },
        )
        return tuple(merged)

    async def __pixel_overlay_observation(
        self,
        *,
        capture: ScreenCapture,
        budget: PerceptionBudget,
        existing: Tuple[OverlayObservation, ...],
        elements: Tuple[PerceivedElement, ...],
    ) -> Optional[OverlayObservation]:
        """
        Build an OverlayObservation from pixel-level evidence when no element-level overlay exists.
        """

        if existing:
            return None

        if (bounds := await self.__pixel_overlay.detect(capture=capture, budget=budget)) is None:
            return None

        candidates = self.__overlay_candidates(elements=elements)
        logger.info(
            "Pixel overlay surfaced into observation",
            extra={
                **self.__log_context(activity=capture.activity),
                "event": "observation.overlay.pixel.surfaced",
                "candidate.count": len(candidates),
            },
        )
        return OverlayObservation(visible=True, bounds=bounds, candidates=candidates)

    @staticmethod
    def __manifest_text_coverage(*, elements: Tuple[PerceivedElement, ...]) -> float:
        """
        Return the fraction of perceived elements that already carry visible text.
        """

        if not elements:
            return 0.0

        with_text = sum(1 for element in elements if element.text)
        return with_text / len(elements)

    def __overlaps_existing(
        self,
        *,
        bounds: Bounds,
        existing: Tuple[PerceivedElement, ...],
    ) -> bool:
        """
        Return whether a candidate bounds duplicates an existing perceived element.
        """

        return any(
            self.__iou(first=bounds, second=element.bounds) >= VISUAL_CONTROL_MINIMUM_IOU
            for element in existing
        )

    def __element_from_token(self, *, token: OcrToken, index: int) -> PerceivedElement:
        """
        Convert one OCR token into a perceived element.
        """

        return PerceivedElement(
            identifier=f"ocr_{index}",
            label_id=str(index),
            text=token.text,
            parent=None,
            bounds=token.bounds,
            source=ElementSource.OCR,
            role=ElementRole.TEXT,
            confidence=token.raw_score,
            tappable=False,
        )

    async def __icon_elements(
        self,
        *,
        capture: ScreenCapture,
        budget: PerceptionBudget,
        existing: Tuple[PerceivedElement, ...],
        start: int,
    ) -> Tuple[PerceivedElement, ...]:
        """
        Call the icon detector port and convert matches into perceived elements.
        """

        result = await self.__icons.detect(capture=capture, budget=budget)
        if not result.matches:
            return ()

        merged: List[PerceivedElement] = []
        for offset, match in enumerate(result.matches):
            if self.__overlaps_existing(bounds=match.bounds, existing=existing):
                continue
            merged.append(self.__element_from_icon(match=match, index=start + offset))
        if merged:
            logger.info(
                "Icon matches merged into observation",
                extra={
                    **self.__log_context(activity=capture.activity),
                    "event": "observation.icon.merged",
                    "merged.count": len(merged),
                    "duration.ms": result.duration,
                },
            )
        return tuple(merged)

    def __log_context(self, *, activity: str) -> Dict[str, Any]:
        """
        Return shared structured-logging context for observation entries.
        """

        return {
            "component": "core.observation",
            "workflow.id": self.__workflow_id,
            "activity": activity,
        }

    def __element_from_icon(self, *, match: IconMatch, index: int) -> PerceivedElement:
        """
        Convert one icon match into a perceived element.
        """

        return PerceivedElement(
            identifier=f"icon_{index}",
            label_id=str(index),
            text=match.kind.value,
            parent=None,
            bounds=match.bounds,
            source=ElementSource.ICON,
            role=ElementRole.ICON,
            confidence=match.confidence,
            tappable=True,
        )

    def __element_from_label(self, *, element: LabeledElement, index: int) -> PerceivedElement:
        """
        Convert a labeled element into a perceived element.
        """

        attributes = element.attributes

        text = self.__text(attributes=attributes)
        bounds = self.__bounds(bounds=element.bounds)
        role = self.__role(attributes=attributes)
        confidence = self.__confidence(value=attributes.get("confidence"))
        source = self.__source(value=str(attributes.get("source", "")).strip())

        identifier = element.label or str(index)
        return PerceivedElement(
            role=role,
            text=text,
            parent=None,
            bounds=bounds,
            source=source,
            confidence=confidence,
            identifier=identifier,
            label_id=identifier,
            tappable=self.__is_tappable(role=role, attributes=attributes),
            scrollable=self.__scrollable(attributes=attributes, role=role),
            axis=self.__axis(attributes=attributes),
            kind=self.__element_kind(attributes=attributes, role=role),
        )

    def __visual_controls(
        self,
        *,
        start: int,
        capture: ScreenCapture,
        existing: Tuple[PerceivedElement, ...],
    ) -> Tuple[PerceivedElement, ...]:
        """
        Detect local screenshot-only visual controls without paid model calls.
        """

        if not capture.image:
            return ()

        if cv2 is None:
            return ()

        image_array = numpy.frombuffer(capture.image, dtype=numpy.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if image is None:
            return ()

        height, width = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = (
            (hsv[:, :, 1] > VISUAL_CONTROL_MINIMUM_SATURATION)
            & (hsv[:, :, 2] > VISUAL_CONTROL_MINIMUM_VALUE)
        ).astype(numpy.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        controls: List[PerceivedElement] = []
        for index in range(1, count):
            x, y, candidate_width, candidate_height, area = (int(value) for value in stats[index])
            bounds = Bounds(
                x=max(0, x),
                y=max(0, y),
                width=max(1, candidate_width),
                height=max(1, candidate_height),
                source=CoordinateSource.VIEWPORT,
                coordinate_system=CoordinateSystem.DEVICE_PIXEL,
            )
            if not self.__viable_visual_control(
                bounds=bounds,
                area=area,
                screen_width=width,
                screen_height=height,
            ):
                continue
            if any(
                self.__iou(first=bounds, second=element.bounds) >= VISUAL_CONTROL_MINIMUM_IOU
                for element in existing
            ):
                continue
            cv_index = start + len(controls)
            controls.append(
                PerceivedElement(
                    identifier=f"cv_{cv_index}",
                    label_id=str(cv_index),
                    bounds=bounds,
                    source=ElementSource.CV,
                    role=ElementRole.BUTTON,
                    confidence=VISUAL_CONTROL_CONFIDENCE,
                    text=None,
                    tappable=True,
                    parent=None,
                )
            )

        return tuple(controls)

    def __viable_visual_control(
        self,
        *,
        area: int,
        bounds: Bounds,
        screen_width: int,
        screen_height: int,
    ) -> bool:
        """
        Return whether a CV region is a plausible tappable control.
        """

        if bounds.width < VISUAL_CONTROL_MINIMUM_WIDTH:
            return False

        if bounds.height < VISUAL_CONTROL_MINIMUM_HEIGHT:
            return False

        if bounds.width > screen_width * VISUAL_CONTROL_MAXIMUM_WIDTH_RATIO:
            return False

        if bounds.height > screen_height * VISUAL_CONTROL_MAXIMUM_HEIGHT_RATIO:
            return False

        if area < VISUAL_CONTROL_MINIMUM_AREA:
            return False

        fill = area / max(1.0, bounds.width * bounds.height)

        return fill >= VISUAL_CONTROL_MINIMUM_FILL_RATIO

    def __source(self, *, value: str) -> ElementSource:
        """
        Normalize element source metadata.
        """

        normalized = value.lower()

        if normalized == "cv":
            return ElementSource.CV

        if normalized == "ocr":
            return ElementSource.OCR

        if normalized == "xml":
            return ElementSource.XML

        if normalized == "model":
            return ElementSource.MODEL

        if normalized == "icon":
            return ElementSource.ICON

        return ElementSource.ACCESSIBILITY

    def __role(self, *, attributes: Dict[str, object]) -> ElementRole:
        """
        Infer a coarse element role from provider metadata.
        """

        kind = self.__kind(attributes=attributes)

        if any(hint in kind for hint in BUTTON_CLASS_HINTS):
            return ElementRole.BUTTON

        if any(hint in kind for hint in INPUT_CLASS_HINTS):
            return ElementRole.INPUT

        if any(hint in kind for hint in KEYBOARD_CLASS_HINTS):
            return ElementRole.KEYBOARD

        if any(hint in kind for hint in SCROLL_CLASS_HINTS):
            return ElementRole.SCROLL_REGION

        if self.__text(attributes=attributes):
            return ElementRole.TEXT

        return ElementRole.UNKNOWN

    def __keyboard(
        self,
        *,
        capture: ScreenCapture,
        elements: Tuple[PerceivedElement, ...],
    ) -> KeyboardObservation:
        """
        Detect visible keyboard state from perceived elements.
        """

        if not self.__configuration.keyboard.enabled:
            return KeyboardObservation(visible=False)

        candidates = tuple(element for element in elements if element.role == ElementRole.KEYBOARD)
        if candidates:
            return KeyboardObservation(visible=True, bounds=candidates[0].bounds, dismiss=())

        visual_keyboard = self.__visual_keyboard(capture=capture)
        if visual_keyboard is not None:
            return KeyboardObservation(visible=True, bounds=visual_keyboard, dismiss=())

        return KeyboardObservation(visible=False)

    def __visual_keyboard(self, *, capture: ScreenCapture) -> Optional[Bounds]:
        """
        Detect a keyboard-like bottom grid from screenshot pixels.
        """

        if self.__uses_ios_hierarchy(capture=capture):
            return None

        if not capture.image:
            return None

        if cv2 is None:
            return None

        image_array = numpy.frombuffer(capture.image, dtype=numpy.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_GRAYSCALE)
        if image is None:
            return None

        height, width = image.shape[:2]
        top = int(height * (1.0 - KEYBOARD_BOTTOM_REGION_RATIO))
        crop = image[top:height, 0:width]

        if crop.size == 0:
            return None

        edges = cv2.Canny(crop, 60, 160)
        edge_density = float(numpy.count_nonzero(edges)) / float(max(1, edges.size))
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if (
            edge_density < KEYBOARD_EDGE_DENSITY_THRESHOLD
            or len(contours) < KEYBOARD_MINIMUM_CONTOUR_COUNT
        ):
            return None

        return Bounds(
            x=0,
            y=top,
            width=width,
            height=height - top,
            source=CoordinateSource.VIEWPORT,
            coordinate_system=CoordinateSystem.DEVICE_PIXEL,
        )

    @staticmethod
    def __uses_ios_hierarchy(*, capture: ScreenCapture) -> bool:
        """
        Return whether the capture was produced from an iOS hierarchy dump.
        """

        xml_content = capture.xml_content or ""
        return "XCUIElementType" in xml_content

    def __overlays(
        self,
        *,
        capture: ScreenCapture,
        elements: Tuple[PerceivedElement, ...],
    ) -> Tuple[OverlayObservation, ...]:
        """
        Return at most one element-level overlay observation per screen.

        Stacked qualifying elements are intentionally collapsed: the supervisor
        only needs a single BLOCKING signal, and surfacing N overlays for N
        layered dialogs produced duplicate downstream effects.
        """

        screen_area = max(1, capture.width * capture.height)
        first = next(
            (
                element
                for element in elements
                if self.__qualifies_as_overlay(element=element, screen_area=screen_area)
            ),
            None,
        )
        if first is None:
            return ()

        return (
            OverlayObservation(
                visible=True,
                bounds=first.bounds,
                candidates=self.__overlay_candidates(elements=elements),
            ),
        )

    @staticmethod
    def __qualifies_as_overlay(*, element: PerceivedElement, screen_area: int) -> bool:
        """
        Whether one perceived element represents a blocking overlay.

        XML/accessibility-sourced elements only qualify when their role is
        explicitly OVERLAY; visual or model-sourced regions qualify on size
        alone. Scroll regions, inputs, and keyboards are never overlays.
        """

        if (
            element.source in {ElementSource.ACCESSIBILITY, ElementSource.XML}
            and element.role != ElementRole.OVERLAY
        ):
            return False

        if element.role in {ElementRole.SCROLL_REGION, ElementRole.INPUT, ElementRole.KEYBOARD}:
            return False

        return (
            element.bounds.width * element.bounds.height
            >= screen_area * OVERLAY_MINIMUM_COVERAGE_RATIO
        )

    def __overlay_candidates(
        self,
        *,
        elements: Tuple[PerceivedElement, ...],
    ) -> Tuple[PerceivedElement, ...]:
        """
        Return actionable candidates that may dismiss an overlay.
        """

        return tuple(
            element
            for element in elements
            if element.tappable and element.role in {ElementRole.BUTTON, ElementRole.ICON}
        )

    def __scroll_regions(
        self,
        *,
        elements: Tuple[PerceivedElement, ...],
        capture: ScreenCapture,
    ) -> Tuple[ScrollRegion, ...]:
        """
        Return scrollable region candidates.
        """

        explicit = tuple(
            ScrollRegion(
                bounds=element.bounds,
                direction="vertical"
                if (element.axis or "vertical") == "vertical"
                else "horizontal",
                confidence=element.confidence,
                identifier=element.identifier,
                label_id=element.label_id,
                observation_region_id=None,
                axis=element.axis or "vertical",
                kind=self.__scope_kind(kind=element.kind, axis=element.axis),
                source=ScrollEvidenceSource.SURFACE,
            )
            for element in elements
            if element.role == ElementRole.SCROLL_REGION
            or element.scrollable
            or self.__is_manifest_scroll_surface_candidate(
                element=element,
                capture=capture,
            )
        )
        explicit = self.__prune_nested_scroll_regions(regions=explicit)
        if explicit:
            large_vertical = tuple(
                region for region in explicit if region.bounds.height >= int(capture.height * 0.35)
            )
            if large_vertical:
                return large_vertical
            if any((region.axis or "vertical") == "horizontal" for region in explicit):
                return explicit

        inferred = self.__page_scroll_region(elements=elements, capture=capture)
        if inferred is None:
            return explicit

        return (*explicit, inferred) if explicit else (inferred,)

    def __page_scroll_region(
        self,
        *,
        elements: Tuple[PerceivedElement, ...],
        capture: ScreenCapture,
    ) -> Optional[ScrollRegion]:
        """
        Infer a page-level vertical scroll lane when XML exposes only nested strips.
        """

        top = self.__page_top_boundary(elements=elements, capture=capture)
        bottom = self.__page_bottom_boundary(elements=elements, capture=capture)
        height = bottom - top
        if height < int(capture.height * 0.30):
            return None

        return ScrollRegion(
            bounds=Bounds(
                x=0,
                y=max(0, top),
                width=capture.width,
                height=min(capture.height - max(0, top), height),
                coordinate_system=CoordinateSystem.DEVICE_PIXEL,
                source=CoordinateSource.VIEWPORT,
            ),
            direction="vertical",
            confidence=0.72,
            identifier="page_scroll_region",
            label_id=None,
            observation_region_id="page_scroll_region",
            axis="vertical",
            kind=CommandScopeKind.VIEWPORT,
            source=ScrollEvidenceSource.SURFACE,
        )

    def __prune_nested_scroll_regions(
        self,
        *,
        regions: Tuple[ScrollRegion, ...],
    ) -> Tuple[ScrollRegion, ...]:
        """
        Drop smaller overlapping fragments when a larger same-axis region already contains them.
        """

        kept: List[ScrollRegion] = []
        for candidate in sorted(
            regions,
            key=lambda region: region.bounds.width * region.bounds.height,
            reverse=True,
        ):
            if any(
                self.__same_axis(first=candidate, second=existing)
                and self.__contains(first=existing.bounds, second=candidate.bounds)
                for existing in kept
            ):
                continue
            kept.append(candidate)
        return tuple(kept)

    @staticmethod
    def __is_manifest_scroll_surface_candidate(
        *,
        element: PerceivedElement,
        capture: ScreenCapture,
    ) -> bool:
        """
        Return whether one manifest-backed container is a plausible scroll surface.
        """

        if element.label_id is None:
            return False
        if element.source not in {ElementSource.XML, ElementSource.ACCESSIBILITY}:
            return False
        if element.tappable:
            return False

        structural_kind = (element.kind or "").lower()
        if element.role not in {
            ElementRole.CONTAINER,
            ElementRole.UNKNOWN,
        } and structural_kind not in {"cell", "container", "list", "other"}:
            return False

        return element.bounds.width >= int(capture.width * 0.80) and element.bounds.height >= int(
            capture.height * 0.30
        )

    @staticmethod
    def __same_axis(*, first: ScrollRegion, second: ScrollRegion) -> bool:
        """
        Return whether two regions describe the same movement axis.
        """

        return (first.axis or "vertical") == (second.axis or "vertical")

    @staticmethod
    def __contains(*, first: Bounds, second: Bounds) -> bool:
        """
        Return whether the first bounds fully contain the second.
        """

        return (
            first.x <= second.x
            and first.y <= second.y
            and first.x + first.width >= second.x + second.width
            and first.y + first.height >= second.y + second.height
        )

    def __page_top_boundary(
        self,
        *,
        elements: Tuple[PerceivedElement, ...],
        capture: ScreenCapture,
    ) -> int:
        """
        Return the safe top boundary for a feed-like page scroll.
        """

        default_top = int(capture.height * 0.15)
        inputs = [
            element
            for element in elements
            if element.role == ElementRole.INPUT
            and element.bounds.y + element.bounds.height <= int(capture.height * 0.45)
        ]
        if not inputs:
            return default_top

        return max(
            default_top,
            max(element.bounds.y + element.bounds.height for element in inputs) + 24,
        )

    @staticmethod
    def __scrollable(*, attributes: Dict[str, object], role: ElementRole) -> bool:
        """
        Return whether one manifest element explicitly represents a scrollable candidate.
        """

        raw = str(attributes.get("scrollable", "")).lower()
        return raw == "true" or role == ElementRole.SCROLL_REGION

    @staticmethod
    def __axis(*, attributes: Dict[str, object]) -> Optional[str]:
        """
        Return the declared movement axis when available.
        """

        axis = str(attributes.get("axis", "")).strip().lower()
        return axis or None

    @staticmethod
    def __element_kind(*, attributes: Dict[str, object], role: ElementRole) -> Optional[str]:
        """
        Return the declared structural kind when available.
        """

        kind = str(attributes.get("kind", "")).strip().lower()
        if kind:
            return kind
        if role == ElementRole.SCROLL_REGION:
            return "container"
        return None

    @staticmethod
    def __scope_kind(*, kind: Optional[str], axis: Optional[str]) -> CommandScopeKind:
        """
        Map element metadata onto one command scope kind.
        """

        normalized = (kind or "").lower()
        if normalized == "carousel":
            return CommandScopeKind.CAROUSEL
        if normalized == "sheet":
            return CommandScopeKind.SHEET
        if normalized == "list":
            return CommandScopeKind.LIST
        if normalized == "viewport":
            return CommandScopeKind.VIEWPORT
        if normalized == "container":
            return CommandScopeKind.CONTAINER
        if axis == "horizontal":
            return CommandScopeKind.CAROUSEL
        return CommandScopeKind.CONTAINER

    def __page_bottom_boundary(
        self,
        *,
        elements: Tuple[PerceivedElement, ...],
        capture: ScreenCapture,
    ) -> int:
        """
        Return the safe bottom boundary for a feed-like page scroll.
        """

        _ = elements
        return int(capture.height * 0.86)

    def __calls_to_action(
        self,
        *,
        elements: Tuple[PerceivedElement, ...],
    ) -> Tuple[PerceivedElement, ...]:
        """
        Return visible prominent controls.
        """

        return tuple(
            element
            for element in elements
            if element.tappable and self.__is_call_to_action(element=element)
        )

    def __focused(
        self,
        *,
        elements: Tuple[PerceivedElement, ...],
    ) -> Optional[PerceivedElement]:
        """
        Return the focused element when provider metadata exposes one.
        """

        for element in elements:
            if element.role == ElementRole.INPUT:
                return element

        return None

    def __is_call_to_action(self, *, element: PerceivedElement) -> bool:
        """
        Return whether an element is a prominent action control.
        """

        text = (element.text or "").lower()
        if any(marker in text for marker in CALL_TO_ACTION_TEXT):
            return True

        return (
            element.role == ElementRole.BUTTON
            and element.bounds.width * element.bounds.height >= CALL_TO_ACTION_MINIMUM_AREA
        )

    def __is_tappable(self, *, role: ElementRole, attributes: Dict[str, object]) -> bool:
        """
        Return whether an element is actionable.
        """

        clickable = str(attributes.get("clickable", "")).strip().lower()
        if clickable == "true":
            return True

        return role in {ElementRole.BUTTON, ElementRole.ICON, ElementRole.INPUT}

    def __bounds(self, *, bounds: UIBounds) -> Bounds:
        """
        Convert UI bounds into action bounds.
        """

        x = max(0, int(round(bounds.x1)))
        y = max(0, int(round(bounds.y1)))
        width = max(1, int(round(bounds.x2 - bounds.x1)))
        height = max(1, int(round(bounds.y2 - bounds.y1)))

        return Bounds(
            x=x,
            y=y,
            width=width,
            height=height,
            coordinate_system=CoordinateSystem.DEVICE_PIXEL,
        )

    def __text(self, *, attributes: Dict[str, object]) -> Optional[str]:
        """
        Return normalized visible text from provider metadata.
        """

        for key in ("text", "label", "name", "content-desc", "value"):
            value = str(attributes.get(key, "")).strip()
            if value:
                return value

        return None

    def __kind(self, *, attributes: Dict[str, object]) -> str:
        """
        Return normalized provider class metadata.
        """

        values = (
            str(attributes.get("class", "")),
            str(attributes.get("type", "")),
            str(attributes.get("role", "")),
        )
        return " ".join(value.lower() for value in values if value)

    def __confidence(self, *, value: Any) -> float:
        """
        Return provider confidence clamped to the valid range.
        """

        try:
            confidence = float(value)
        except (TypeError, ValueError):
            confidence = 1.0

        return max(0.0, min(1.0, confidence))

    @staticmethod
    def __iou(*, first: Bounds, second: Bounds) -> float:
        """
        Return intersection-over-union for two bounds.
        """

        top = max(first.y, second.y)
        left = max(first.x, second.x)
        right = min(first.x + first.width, second.x + second.width)
        bottom = min(first.y + first.height, second.y + second.height)

        if right <= left or bottom <= top:
            return 0.0

        intersection = (right - left) * (bottom - top)

        first_area = first.width * first.height
        second_area = second.width * second.height
        union = first_area + second_area - intersection

        if union <= 0:
            return 0.0

        return float(intersection / union)
