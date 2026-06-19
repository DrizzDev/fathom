from __future__ import annotations

import asyncio
import time
from logging import getLogger
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy

from fathom.constants.perception import (
    ICON_MATCH_MINIMUM_SCORE,
    ICON_NON_MAX_SUPPRESSION_IOU,
)
from fathom.interfaces.icon import IconDetectorPort
from fathom.schemas.actions import Bounds, CoordinateSource, CoordinateSystem
from fathom.schemas.budgets import PerceptionBudget
from fathom.schemas.icon import IconDetectionResult, IconKind, IconMatch, IconTemplate
from fathom.schemas.screens import ScreenCapture

logger = getLogger(__name__)


class TemplateIconDetector(IconDetectorPort):
    """
    Icon detector that runs OpenCV template matching against a bundled registry.
    """

    def __init__(
        self,
        *,
        workflow_id: Optional[str] = None,
        templates: Tuple[IconTemplate, ...] = (),
        minimum_score: float = ICON_MATCH_MINIMUM_SCORE,
        suppression_iou: float = ICON_NON_MAX_SUPPRESSION_IOU,
    ) -> None:
        """
        Initialize the detector with a frozen template registry and scoring thresholds.
        """

        self.__workflow_id = workflow_id
        self.__minimum_score = minimum_score
        self.__suppression_iou = suppression_iou
        self.__compiled = self.__compile_templates(templates=templates)

    async def detect(
        self, *, capture: ScreenCapture, budget: PerceptionBudget
    ) -> IconDetectionResult:
        """
        Run template matching; degrade to an empty result on any failure.
        """

        if not self.__compiled or not capture.image:
            return IconDetectionResult(matches=(), duration=0)

        started = time.monotonic()
        timeout = budget.local / 1000.0
        context = self.__log_context(activity=capture.activity)

        try:
            matches = await asyncio.wait_for(
                asyncio.to_thread(self.__match_all, capture.image),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Icon detector timed out — degrading to no matches",
                extra={
                    **context,
                    "event": "icon.detect.timeout",
                    "budget.local.ms": budget.local,
                },
            )
            return IconDetectionResult(
                matches=(),
                duration=int((time.monotonic() - started) * 1000),
            )
        except Exception:
            # Icon template matching is an optional perception
            # enrichment; cv2 / numpy failures must not break the run.
            logger.exception(
                "Icon detector failed — degrading to no matches",
                extra={
                    **context,
                    "event": "icon.detect.failed",
                    "budget.local.ms": budget.local,
                },
            )
            return IconDetectionResult(
                matches=(),
                duration=int((time.monotonic() - started) * 1000),
            )

        duration = int((time.monotonic() - started) * 1000)

        logger.info(
            "Icon detector completed",
            extra={
                **context,
                "duration.ms": duration,
                "match.count": len(matches),
                "event": "icon.detect.completed",
            },
        )
        return IconDetectionResult(matches=tuple(matches), duration=duration)

    def __match_all(self, image: bytes) -> List[IconMatch]:
        """
        Decode the screenshot once and match every compiled template against it.
        """

        screen_array = numpy.frombuffer(image, dtype=numpy.uint8)
        if (screen_gray := cv2.imdecode(screen_array, cv2.IMREAD_GRAYSCALE)) is None:
            return []

        matches: List[IconMatch] = []
        for kind, template in self.__compiled:
            matches.extend(
                self.__match_one(
                    kind=kind,
                    template=template,
                    screen=screen_gray,
                )
            )
        return self.__suppress(matches=matches)

    def __match_one(
        self,
        *,
        kind: IconKind,
        screen: numpy.ndarray,
        template: numpy.ndarray,
    ) -> List[IconMatch]:
        """
        Run normalized template matching for one template and surface peak matches.
        """

        if template.shape[0] > screen.shape[0] or template.shape[1] > screen.shape[1]:
            return []

        height, width = template.shape[:2]
        response = cv2.matchTemplate(screen, template, cv2.TM_CCOEFF_NORMED)
        peaks = numpy.argwhere(response >= self.__minimum_score)

        matches: List[IconMatch] = []
        for point in peaks:
            y, x = int(point[0]), int(point[1])
            score = float(response[y, x])
            bounds = Bounds(
                x=x,
                y=y,
                width=int(width),
                height=int(height),
                source=CoordinateSource.VIEWPORT,
                coordinate_system=CoordinateSystem.DEVICE_PIXEL,
            )
            matches.append(IconMatch(kind=kind, bounds=bounds, confidence=score))
        return matches

    def __suppress(self, *, matches: List[IconMatch]) -> List[IconMatch]:
        """
        Drop overlapping candidates per icon kind so only the strongest survives.
        """

        survivors: List[IconMatch] = []
        ordered = sorted(matches, key=lambda match: match.confidence, reverse=True)

        for candidate in ordered:
            if any(
                survivor.kind == candidate.kind
                and self.__iou(first=survivor.bounds, second=candidate.bounds)
                >= self.__suppression_iou
                for survivor in survivors
            ):
                continue
            survivors.append(candidate)

        return survivors

    def __compile_templates(
        self,
        *,
        templates: Tuple[IconTemplate, ...],
    ) -> Tuple[Tuple[IconKind, numpy.ndarray], ...]:
        """
        Decode every template image once at construction time.
        """

        compiled: List[Tuple[IconKind, numpy.ndarray]] = []

        for template in templates:
            buffer = numpy.frombuffer(template.image, dtype=numpy.uint8)
            if (decoded := cv2.imdecode(buffer, cv2.IMREAD_GRAYSCALE)) is None:
                continue

            compiled.append((template.kind, decoded))

        return tuple(compiled)

    @staticmethod
    def __iou(*, first: Bounds, second: Bounds) -> float:
        """
        Return the intersection-over-union for two pixel bounds.
        """

        top = max(first.y, second.y)
        left = max(first.x, second.x)
        right = min(first.x + first.width, second.x + second.width)
        bottom = min(first.y + first.height, second.y + second.height)

        if right <= left or bottom <= top:
            return 0.0

        first_area = first.width * first.height
        second_area = second.width * second.height
        intersection = (right - left) * (bottom - top)
        union = first_area + second_area - intersection

        if union <= 0:
            return 0.0

        return float(intersection / union)

    def __log_context(self, *, activity: str) -> Dict[str, Any]:
        """
        Return shared structured-logging context for icon-detector entries.
        """

        return {
            "activity": activity,
            "workflow.id": self.__workflow_id,
            "component": "adapter.icon.template",
            "templates.compiled": len(self.__compiled),
        }
