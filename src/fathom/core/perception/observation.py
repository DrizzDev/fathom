from __future__ import annotations

import time
from logging import getLogger
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy

from fathom.adapters.icon.noop import NoopIconDetector
from fathom.adapters.ocr.noop import NoopOcr
from fathom.adapters.perception.overlay.noop import NoopOverlayDetector
from fathom.constants.perception import (
    BUTTON_CLASS_HINTS,
    CALL_TO_ACTION_MINIMUM_AREA,
    CALL_TO_ACTION_TEXT,
    INPUT_CLASS_HINTS,
    KEYBOARD_BOTTOM_REGION_RATIO,
    KEYBOARD_CLASS_HINTS,
    KEYBOARD_EDGE_DENSITY_THRESHOLD,
    KEYBOARD_MINIMUM_CONTOUR_COUNT,
    KEYBOARD_MINIMUM_HEIGHT_RATIO,
    KEYBOARD_MINIMUM_TOP_RATIO,
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
        ocr: Optional[OcrPort] = None,
        icons: Optional[IconDetectorPort] = None,
        pixel_overlay: Optional[OverlayDetectorPort] = None,
        pipeline: Optional[ArtifactPipeline] = None,
        workflow_id: Optional[str] = None,
    ) -> None:
        """
        Initialize the observation service with optional providers and run context.
        """

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

        visual_elements = self.__visual_controls(
            capture=capture,
            existing=elements,
            start=len(elements) + 1,
        )
        elements = (*elements, *visual_elements)

        if self.__manifest_text_coverage(elements=elements) < OCR_TRIGGER_MANIFEST_TEXT_COVERAGE:
            ocr_elements = await self.__ocr_elements(
                capture=capture,
                budget=budget,
                existing=elements,
                start=len(elements) + 1,
            )
            elements = (*elements, *ocr_elements)

        icon_elements = await self.__icon_elements(
            capture=capture,
            budget=budget,
            existing=elements,
            start=len(elements) + 1,
        )
        elements = (*elements, *icon_elements)

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

        scroll = self.__scroll_regions(elements=elements)
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

        Saving an OCR-only image when OCR did not run produces a useless
        blank-overlay file that pollutes the asset directory. We gate
        each per-source artifact behind the presence of at least one
        element from that source. The merged-hybrid artifact is only
        emitted when at least one non-XML element exists — XML boxes
        are already on the manifest-annotated image.
        """

        if self.__pipeline is None:
            return

        created = int(time.time() * 1000)
        sources = {element.source for element in observation.elements}

        if any(source is not ElementSource.XML for source in sources):
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

        candidates = tuple(element for element in elements if element.role == ElementRole.KEYBOARD)
        if candidates:
            return KeyboardObservation(visible=True, bounds=candidates[0].bounds, dismiss=())

        lower_bound = int(capture.height * KEYBOARD_MINIMUM_TOP_RATIO)
        minimum_height = int(capture.height * KEYBOARD_MINIMUM_HEIGHT_RATIO)

        bottom_controls = tuple(
            element
            for element in elements
            if element.bounds.y >= lower_bound and element.bounds.height >= minimum_height
        )
        if not bottom_controls:
            visual_keyboard = self.__visual_keyboard(capture=capture)
            if visual_keyboard is not None:
                return KeyboardObservation(visible=True, bounds=visual_keyboard, dismiss=())

            return KeyboardObservation(visible=False)

        return KeyboardObservation(visible=True, bounds=bottom_controls[0].bounds, dismiss=())

    def __visual_keyboard(self, *, capture: ScreenCapture) -> Optional[Bounds]:
        """
        Detect a keyboard-like bottom grid from screenshot pixels.
        """

        if not capture.image:
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
    ) -> Tuple[ScrollRegion, ...]:
        """
        Return scrollable region candidates.
        """

        return tuple(
            ScrollRegion(
                bounds=element.bounds,
                direction="vertical",
                confidence=element.confidence,
            )
            for element in elements
            if element.role == ElementRole.SCROLL_REGION
        )

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
