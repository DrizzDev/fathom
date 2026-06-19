from __future__ import annotations

import asyncio
import importlib
import io
import time
from logging import getLogger
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy
from PIL import Image

from fathom.adapters.icon.noop import NoopIconDetector
from fathom.adapters.ocr.noop import NoopOcr
from fathom.adapters.perception.overlay.noop import NoopOverlayDetector
from fathom.constants.command import CommandScopeKind
from fathom.constants.observation import KeyboardVisibility
from fathom.constants.perception import (
    BUTTON_CLASS_HINTS,
    CALL_TO_ACTION_MINIMUM_AREA,
    CALL_TO_ACTION_TEXT,
    INPUT_CLASS_HINTS,
    KEYBOARD_CLASS_HINTS,
    MAX_ACTION_BOUND,
    OCR_MAXIMUM_TOKEN_LENGTH,
    OCR_TRIGGER_MANIFEST_TEXT_COVERAGE,
    OCR_TRIGGER_MIN_MANIFEST_SIZE,
    OCR_TRIGGER_MIN_TEXT_BEARING_ELEMENTS,
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
from fathom.constants.screen import ScreenKind
from fathom.constants.scroll import ScrollEvidenceSource
from fathom.core.artifact.pipeline import ArtifactPipeline
from fathom.core.exceptions import OcrError
from fathom.interfaces.device import DevicePort
from fathom.interfaces.icon import IconDetectorPort
from fathom.interfaces.ocr import OcrPort
from fathom.interfaces.overlay import OverlayDetectorPort
from fathom.processing.parsers.kind import ScreenKindClassifier
from fathom.schemas.actions import Bounds, CoordinateSource, CoordinateSystem
from fathom.schemas.artifact import (
    ArtifactRecord,
    CvPerceptionPayload,
    IconPerceptionPayload,
    OcrPerceptionPayload,
    OcrRawPayload,
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

try:
    cv2: Any = importlib.import_module("cv2")
except ModuleNotFoundError:  # pragma: no cover - dependency optional when CV is disabled
    cv2 = None

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
        device: Optional[DevicePort] = None,
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
        self.__device = device
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

        manifest_elements: List[PerceivedElement] = []
        capture_system = self.__capture_dimension_system(capture=capture)

        for index, label in enumerate(manifest, start=1):
            element = self.__element_from_label(
                element=label,
                index=index,
                capture=capture,
                capture_system=capture_system,
            )
            if element is not None:
                manifest_elements.append(element)

        elements = tuple(manifest_elements)

        if self.__configuration.cv.enabled:
            if cv2 is None:
                raise RuntimeError("OpenCV is required when perception.cv.enabled is true.")

            visual_elements = self.__visual_controls(
                capture=capture,
                existing=elements,
                start=len(elements) + 1,
            )
            elements = (*elements, *visual_elements)

        screen_kind = ScreenKindClassifier.classify(
            screen_width=capture.width,
            screen_height=capture.height,
            xml_content=capture.xml_content,
        )

        elements = await self.__merge_async_enrichment(
            budget=budget,
            capture=capture,
            elements=elements,
            session_id=session_id,
            step_number=step_number,
            screen_kind=screen_kind,
        )

        keyboard = await self.__keyboard(elements=elements, capture=capture)
        overlays = self.__overlays(elements=elements, capture=capture)

        if (
            pixel_overlay := await self.__pixel_overlay_observation(
                budget=budget,
                capture=capture,
                existing=overlays,
                elements=elements,
            )
        ) is not None:
            overlays = (*overlays, pixel_overlay)

        scroll = self.__scroll_regions(
            capture=capture,
            elements=elements,
            capture_system=capture_system,
        )
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
            session_id=session_id,
            step_number=step_number,
            observation=observation,
        )

        return observation

    async def __merge_async_enrichment(
        self,
        *,
        session_id: str,
        step_number: int,
        capture: ScreenCapture,
        screen_kind: ScreenKind,
        budget: PerceptionBudget,
        elements: Tuple[PerceivedElement, ...],
    ) -> Tuple[PerceivedElement, ...]:
        """
        Merge optional OCR and icon enrichers without serializing their wall time.
        """

        ocr_task = None
        base = elements

        if screen_kind is not ScreenKind.NATIVE or self.__should_run_ocr(elements=base):
            logger.info(
                "OCR budget selected",
                extra={
                    "event": "ocr.budget.selected",
                    "budget.ms": budget.ocr,
                    "screen.kind": screen_kind.value,
                    "reason": (
                        "hierarchy_blind"
                        if screen_kind is not ScreenKind.NATIVE
                        else "low_text_coverage"
                    ),
                    **self.__log_context(activity=capture.activity),
                },
            )
            ocr_task = asyncio.create_task(
                self.__ocr_elements(
                    budget=budget,
                    existing=base,
                    capture=capture,
                    start=len(base) + 1,
                    session_id=session_id,
                    step_number=step_number,
                )
            )
        else:
            logger.info(
                "OCR skipped — hierarchy already provides text coverage",
                extra={
                    "event": "ocr.skipped",
                    "reason": "hierarchy_text_coverage_sufficient",
                    "screen.kind": screen_kind.value,
                    **self.__log_context(activity=capture.activity),
                },
            )

        icon_task = asyncio.create_task(
            self.__icon_elements(
                budget=budget,
                existing=base,
                capture=capture,
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
        session_id: str,
        step_number: int,
        capture: ScreenCapture,
        observation: ScreenObservation,
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
            ElementSource.CV,
            ElementSource.OCR,
            ElementSource.ICON,
            ElementSource.VISION,
        }

        if any(source in enrichment_sources for source in sources):
            await self.__pipeline.emit(
                record=ArtifactRecord(
                    created=created,
                    session_id=session_id,
                    step_number=step_number,
                    package_name=capture.activity,
                    payload=PerceptionPayload(capture=capture, observation=observation),
                ),
            )
        for source, payload_factory in (
            (ElementSource.CV, CvPerceptionPayload),
            (ElementSource.OCR, OcrPerceptionPayload),
            (ElementSource.ICON, IconPerceptionPayload),
            (ElementSource.VISION, VisionPerceptionPayload),
        ):
            if source not in sources:
                continue

            await self.__pipeline.emit(
                record=ArtifactRecord(
                    created=created,
                    session_id=session_id,
                    step_number=step_number,
                    package_name=capture.activity,
                    payload=payload_factory(capture=capture, observation=observation),
                ),
            )
        if observation.overlays:
            await self.__pipeline.emit(
                record=ArtifactRecord(
                    created=created,
                    session_id=session_id,
                    step_number=step_number,
                    package_name=capture.activity,
                    payload=OverlayPerceptionPayload(
                        capture=capture,
                        observation=observation,
                    ),
                ),
            )

    async def __ocr_elements(
        self,
        *,
        start: int,
        session_id: str,
        step_number: int,
        capture: ScreenCapture,
        budget: PerceptionBudget,
        existing: Tuple[PerceivedElement, ...],
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
                    "reason": exception.message,
                    "retryable": exception.retryable,
                    "event": "observation.ocr.skipped",
                    **self.__log_context(activity=capture.activity),
                },
            )
            return ()

        await self.__emit_ocr_raw_artifact(
            capture=capture,
            session_id=session_id,
            step_number=step_number,
            raw_response=result.raw_response,
        )

        tokens = tuple(
            token for token in result.tokens if len(token.text) <= OCR_MAXIMUM_TOKEN_LENGTH
        )
        if not tokens:
            logger.info(
                "OCR returned no usable tokens",
                extra={
                    "event": "observation.ocr.empty",
                    "raw.token.count": len(result.tokens),
                    **self.__log_context(activity=capture.activity),
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
                "merged.count": len(merged),
                "duration.ms": result.duration,
                "event": "observation.ocr.merged",
                **self.__log_context(activity=capture.activity),
            },
        )
        return tuple(merged)

    async def __emit_ocr_raw_artifact(
        self,
        *,
        session_id: str,
        step_number: int,
        capture: ScreenCapture,
        raw_response: Optional[str],
    ) -> None:
        """
        Persist raw OCR provider JSON next to XML hierarchy artifacts.
        """

        if self.__pipeline is None or not raw_response:
            return

        await self.__pipeline.emit(
            record=ArtifactRecord(
                session_id=session_id,
                step_number=step_number,
                package_name=capture.activity,
                created=int(time.time() * 1000),
                payload=OcrRawPayload(content=raw_response),
            ),
        )

    async def __pixel_overlay_observation(
        self,
        *,
        capture: ScreenCapture,
        budget: PerceptionBudget,
        elements: Tuple[PerceivedElement, ...],
        existing: Tuple[OverlayObservation, ...],
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
                "candidate.count": len(candidates),
                "event": "observation.overlay.pixel.surfaced",
                **self.__log_context(activity=capture.activity),
            },
        )
        return OverlayObservation(visible=True, bounds=bounds, candidates=candidates)

    @staticmethod
    def __manifest_text_coverage(*, elements: Tuple[PerceivedElement, ...]) -> float:
        """
        Return the fraction of perceived elements that already carry text labels.
        """

        if not elements:
            return 0.0

        with_text = sum(1 for element in elements if element.text)
        return with_text / len(elements)

    @classmethod
    def __should_run_ocr(cls, *, elements: Tuple[PerceivedElement, ...]) -> bool:
        """
        Run OCR unless the manifest is rich on all three axes (size, text-bearing
        count, coverage). Skipping when needed blinds the planner; running when
        redundant costs seconds — so the gate biases toward running.
        """

        if len(elements) < OCR_TRIGGER_MIN_MANIFEST_SIZE:
            return True

        text_bearing = sum(1 for element in elements if element.text)
        if text_bearing < OCR_TRIGGER_MIN_TEXT_BEARING_ELEMENTS:
            return True

        coverage = cls.__manifest_text_coverage(elements=elements)
        return coverage < OCR_TRIGGER_MANIFEST_TEXT_COVERAGE

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
        The Document AI layout level is woven into the identifier so production traces attribute snaps to the level the merge structure came from.
        """

        return PerceivedElement(
            parent=None,
            tappable=False,
            text=token.text,
            label_id=str(index),
            bounds=token.bounds,
            role=ElementRole.TEXT,
            source=ElementSource.OCR,
            confidence=token.raw_score,
            identifier=f"ocr__{token.level.value.lower()}__{index}",
        )

    async def __icon_elements(
        self,
        *,
        start: int,
        capture: ScreenCapture,
        budget: PerceptionBudget,
        existing: Tuple[PerceivedElement, ...],
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
                    "merged.count": len(merged),
                    "duration.ms": result.duration,
                    "event": "observation.icon.merged",
                    **self.__log_context(activity=capture.activity),
                },
            )
        return tuple(merged)

    def __log_context(self, *, activity: str) -> Dict[str, Any]:
        """
        Return shared structured-logging context for observation entries.
        """

        return {
            "activity": activity,
            "component": "core.observation",
            "workflow.id": self.__workflow_id,
        }

    def __element_from_icon(self, *, match: IconMatch, index: int) -> PerceivedElement:
        """
        Convert one icon match into a perceived element.
        """

        return PerceivedElement(
            parent=None,
            tappable=True,
            label_id=str(index),
            bounds=match.bounds,
            text=match.kind.value,
            role=ElementRole.ICON,
            source=ElementSource.ICON,
            identifier=f"icon_{index}",
            confidence=match.confidence,
        )

    def __element_from_label(
        self,
        *,
        index: int,
        capture: ScreenCapture,
        element: LabeledElement,
        capture_system: CoordinateSystem,
    ) -> Optional[PerceivedElement]:
        """
        Convert a labeled element into a perceived element.
        """

        attributes = element.attributes

        text = self.__text(attributes=attributes)
        bounds = self.__bounds(
            bounds=element.bounds,
            capture=capture,
            attributes=attributes,
            capture_system=capture_system,
        )
        if bounds is None:
            return None

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
            label_id=identifier,
            confidence=confidence,
            identifier=identifier,
            axis=self.__axis(attributes=attributes),
            kind=self.__element_kind(attributes=attributes, role=role),
            tappable=self.__is_tappable(role=role, attributes=attributes),
            scrollable=self.__scrollable(attributes=attributes, role=role),
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

        mask = cast(
            "Any",
            cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2),
        )
        mask = cast(
            "Any",
            cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1),
        )

        controls: List[PerceivedElement] = []
        count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)

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
                area=area,
                bounds=bounds,
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
                    text=None,
                    parent=None,
                    tappable=True,
                    bounds=bounds,
                    label_id=str(cv_index),
                    source=ElementSource.CV,
                    role=ElementRole.BUTTON,
                    identifier=f"CV__{cv_index}",
                    confidence=VISUAL_CONTROL_CONFIDENCE,
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

    async def __keyboard(
        self,
        *,
        capture: ScreenCapture,
        elements: Tuple[PerceivedElement, ...],
    ) -> KeyboardObservation:
        """
        Detect keyboard state from perceived elements, falling back to the device IME probe.
        """

        _ = capture

        if self.__configuration.keyboard.enabled:
            candidates = tuple(
                element for element in elements if element.role == ElementRole.KEYBOARD
            )
            if candidates:
                return KeyboardObservation(
                    dismiss=(),
                    bounds=candidates[0].bounds,
                    visibility=KeyboardVisibility.VISIBLE,
                )

        if self.__device is None:
            return KeyboardObservation(visibility=KeyboardVisibility.UNKNOWN)

        try:
            return await self.__device.detect_keyboard()
        except Exception as exception:
            logger.warning(
                "Device keyboard probe failed; treating as UNKNOWN",
                extra={
                    "error": str(exception),
                    "component": "core.perception.observation",
                    "event": "observation.keyboard.probe.failed",
                },
            )
            return KeyboardObservation(visibility=KeyboardVisibility.UNKNOWN)

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

        XML/accessibility-sourced elements only qualify when their role is explicitly OVERLAY;
        visual or model-sourced regions qualify on size alone. Scroll regions, inputs, and keyboards are never overlays.
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
        capture: ScreenCapture,
        capture_system: CoordinateSystem,
        elements: Tuple[PerceivedElement, ...],
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
                label_id=element.label_id,
                observation_region_id=None,
                confidence=element.confidence,
                identifier=element.identifier,
                axis=element.axis or "vertical",
                source=ScrollEvidenceSource.SURFACE,
                kind=self.__scope_kind(kind=element.kind, axis=element.axis),
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

        inferred = self.__page_scroll_region(
            capture=capture,
            elements=elements,
            capture_system=capture_system,
        )
        if inferred is None:
            return explicit

        return (*explicit, inferred) if explicit else (inferred,)

    def __page_scroll_region(
        self,
        *,
        capture: ScreenCapture,
        capture_system: CoordinateSystem,
        elements: Tuple[PerceivedElement, ...],
    ) -> Optional[ScrollRegion]:
        """
        Infer a page-level vertical scroll lane when XML exposes only nested strips.
        """

        top = self.__page_top_boundary(elements=elements, capture=capture)
        bottom = self.__page_bottom_boundary(elements=elements, capture=capture)

        height = bottom - top
        if height < int(capture.height * 0.30):
            logger.warning(
                "Skipped inferred page scroll region because visible lane is too small",
                extra={
                    "region.top": top,
                    "region.bottom": bottom,
                    "region.height": height,
                    "component": "core.observation",
                    "capture.width": capture.width,
                    "capture.height": capture.height,
                    "reason": "height_below_threshold",
                    "capture.system": capture_system.value,
                    "event": "scroll.region.inferred.skipped",
                },
            )
            return None

        region = ScrollRegion(
            bounds=Bounds(
                x=0,
                y=max(0, top),
                width=capture.width,
                coordinate_system=capture_system,
                source=CoordinateSource.VIEWPORT,
                height=min(capture.height - max(0, top), height),
            ),
            label_id=None,
            confidence=0.72,
            axis="vertical",
            direction="vertical",
            identifier="page_scroll_region",
            kind=CommandScopeKind.VIEWPORT,
            source=ScrollEvidenceSource.SURFACE,
            observation_region_id="page_scroll_region",
        )

        logger.info(
            "Inferred page scroll region",
            extra={
                "component": "core.observation",
                "event": "scroll.region.inferred",
                "region.axis": region.axis,
                "bounds.x": region.bounds.x,
                "bounds.y": region.bounds.y,
                "region.id": region.identifier,
                "capture.width": capture.width,
                "capture.height": capture.height,
                "region.kind": region.kind.value,
                "bounds.width": region.bounds.width,
                "region.direction": region.direction,
                "bounds.height": region.bounds.height,
                "capture.system": capture_system.value,
                "region.confidence": region.confidence,
                "bounds.system": region.bounds.system.value,
                "bounds.source": region.bounds.source.value if region.bounds.source else None,
            },
        )
        return region

    @staticmethod
    def __capture_dimension_system(*, capture: ScreenCapture) -> CoordinateSystem:
        """
        Return the coordinate system represented by ``capture.width`` / ``height``.
        """

        if not capture.image:
            logger.info(
                "Capture dimensions classified without image bytes",
                extra={
                    "component": "core.observation",
                    "reason": "missing_image",
                    "capture.width": capture.width,
                    "capture.height": capture.height,
                    "event": "capture.dimension_system.detected",
                    "capture.system": CoordinateSystem.DEVICE_PIXEL.value,
                },
            )
            return CoordinateSystem.DEVICE_PIXEL

        try:
            with Image.open(io.BytesIO(capture.image)) as image:
                if image.width != capture.width or image.height != capture.height:
                    logger.info(
                        "Capture dimensions classified as logical",
                        extra={
                            "component": "core.observation",
                            "reason": "image_size_mismatch",
                            "image.width": image.width,
                            "image.height": image.height,
                            "capture.width": capture.width,
                            "capture.height": capture.height,
                            "event": "capture.dimension_system.detected",
                            "capture.system": CoordinateSystem.LOGICAL.value,
                        },
                    )
                    return CoordinateSystem.LOGICAL

                logger.info(
                    "Capture dimensions classified as device pixels",
                    extra={
                        "component": "core.observation",
                        "reason": "image_size_match",
                        "image.width": image.width,
                        "image.height": image.height,
                        "capture.width": capture.width,
                        "capture.height": capture.height,
                        "event": "capture.dimension_system.detected",
                        "capture.system": CoordinateSystem.DEVICE_PIXEL.value,
                    },
                )
        except Exception:
            logger.warning(
                "Could not inspect capture image dimensions; treating capture dimensions as device pixels",
                extra={
                    "component": "core.observation",
                    "event": "capture.dimension_system.unknown",
                },
            )

        return CoordinateSystem.DEVICE_PIXEL

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
            reverse=True,
            key=lambda region: region.bounds.width * region.bounds.height,
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
            ElementRole.UNKNOWN,
            ElementRole.CONTAINER,
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
        capture: ScreenCapture,
        elements: Tuple[PerceivedElement, ...],
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

    def __bounds(
        self,
        *,
        bounds: UIBounds,
        capture: ScreenCapture,
        attributes: Dict[str, object],
        capture_system: CoordinateSystem,
    ) -> Optional[Bounds]:
        """
        Convert UI bounds into viewport-clipped action bounds.
        """

        viewport_width = max(1, min(MAX_ACTION_BOUND, int(capture.width)))
        viewport_height = max(1, min(MAX_ACTION_BOUND, int(capture.height)))

        x1 = max(0, min(viewport_width, int(round(bounds.x1))))
        y1 = max(0, min(viewport_height, int(round(bounds.y1))))

        x2 = max(0, min(viewport_width, int(round(bounds.x2))))
        y2 = max(0, min(viewport_height, int(round(bounds.y2))))

        if x2 <= x1 or y2 <= y1:
            logger.warning(
                "Dropping element outside viewport",
                extra={
                    "bounds": bounds.model_dump(),
                    "viewport.width": viewport_width,
                    "viewport.height": viewport_height,
                    "event": "observation.bounds.dropped",
                    **self.__log_context(activity=capture.activity),
                },
            )
            return None

        return Bounds(
            x=x1,
            y=y1,
            width=max(1, x2 - x1),
            height=max(1, y2 - y1),
            coordinate_system=self.__element_bounds_system(
                attributes=attributes,
                capture_system=capture_system,
            ),
        )

    @staticmethod
    def __element_bounds_system(
        *,
        attributes: Dict[str, object],
        capture_system: CoordinateSystem,
    ) -> CoordinateSystem:
        """
        Return the coordinate system carried by a manifest element's bounds.
        """

        if "logical_bounds" in attributes:
            return CoordinateSystem.DEVICE_PIXEL

        return capture_system

    def __text(self, *, attributes: Dict[str, object]) -> Optional[str]:
        """
        Return normalized visible text from provider metadata.
        """

        for key in ("text", "label", "name", "content-desc", "value"):
            if value := str(attributes.get(key, "")).strip():
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
