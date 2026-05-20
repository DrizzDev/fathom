from __future__ import annotations

import asyncio
import time
from logging import getLogger
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy

from fathom.constants.perception import (
    PIXEL_OVERLAY_MAX_INTENSITY,
    PIXEL_OVERLAY_MAX_VARIANCE,
    PIXEL_OVERLAY_MIN_AREA_RATIO,
)
from fathom.interfaces.overlay import OverlayDetectorPort
from fathom.schemas.actions import Bounds, CoordinateSource, CoordinateSystem
from fathom.schemas.budgets import PerceptionBudget
from fathom.schemas.screens import ScreenCapture

logger = getLogger(__name__)


class PixelOverlayDetector(OverlayDetectorPort):
    """
    Detects large dim/scrim regions in the screen capture using OpenCV.
    """

    def __init__(
        self,
        *,
        maximum_intensity: int = PIXEL_OVERLAY_MAX_INTENSITY,
        minimum_area_ratio: float = PIXEL_OVERLAY_MIN_AREA_RATIO,
        maximum_variance: float = PIXEL_OVERLAY_MAX_VARIANCE,
        workflow_id: Optional[str] = None,
    ) -> None:
        """
        Initialize the detector with intensity, area, and variance thresholds.
        """

        self.__maximum_intensity = maximum_intensity
        self.__minimum_area_ratio = minimum_area_ratio
        self.__maximum_variance = maximum_variance
        self.__workflow_id = workflow_id

    async def detect(
        self,
        *,
        capture: ScreenCapture,
        budget: PerceptionBudget,
    ) -> Optional[Bounds]:
        """
        Return the dim-overlay bounds when a scrim region passes every threshold.
        """

        if not capture.image:
            return None

        started = time.monotonic()
        timeout = max(0.001, budget.local / 1000.0)
        context = self.__log_context(activity=capture.activity)

        try:
            bounds = await asyncio.wait_for(
                asyncio.to_thread(self.__detect_sync, capture.image),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Pixel overlay timed out — degrading to no overlay",
                extra={
                    **context,
                    "event": "overlay.pixel.timeout",
                    "budget.local.ms": budget.local,
                },
            )
            return None
        except Exception:
            # Pixel overlay is an optional perception enrichment;
            # cv2 / numpy / decode failures must not break the run.
            logger.exception(
                "Pixel overlay detection failed — degrading to no overlay",
                extra={
                    **context,
                    "event": "overlay.pixel.failed",
                    "budget.local.ms": budget.local,
                },
            )
            return None

        duration = int((time.monotonic() - started) * 1000)

        if bounds is not None:
            logger.info(
                "Pixel overlay detected",
                extra={
                    **context,
                    "event": "overlay.pixel.detected",
                    "duration.ms": duration,
                    "coverage.ratio": self.__coverage_ratio(
                        bounds=bounds,
                        width=capture.width,
                        height=capture.height,
                    ),
                },
            )
        return bounds

    def __detect_sync(self, image: bytes) -> Optional[Bounds]:
        """
        Run the synchronous OpenCV pipeline that classifies one screen capture.
        """

        buffer = numpy.frombuffer(image, dtype=numpy.uint8)
        if (gray := cv2.imdecode(buffer, cv2.IMREAD_GRAYSCALE)) is None:
            return None

        height, width = gray.shape[:2]
        if (component := self.__largest_dim_component(gray=gray)) is None:
            return None

        x, y, component_width, component_height, area = component
        area_ratio = area / max(1.0, float(width * height))
        if area_ratio < self.__minimum_area_ratio:
            return None

        region = gray[y : y + component_height, x : x + component_width]
        if region.size == 0:
            return None

        variance = float(numpy.var(region))
        if variance > self.__maximum_variance:
            return None

        return Bounds(
            x=int(x),
            y=int(y),
            source=CoordinateSource.VIEWPORT,
            width=int(max(1, component_width)),
            height=int(max(1, component_height)),
            coordinate_system=CoordinateSystem.DEVICE_PIXEL,
        )

    def __largest_dim_component(
        self,
        *,
        gray: numpy.ndarray,
    ) -> Optional[Tuple[int, int, int, int, int]]:
        """
        Return the largest connected dim component as (x, y, width, height, area).
        """

        _, mask = cv2.threshold(
            gray,
            self.__maximum_intensity,
            255,
            cv2.THRESH_BINARY_INV,
        )
        count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        if count <= 1:
            return None

        best_index = 0
        best_area = 0
        for index in range(1, count):
            area = int(stats[index, cv2.CC_STAT_AREA])
            if area > best_area:
                best_area = area
                best_index = index
        if best_index == 0:
            return None

        return (
            int(stats[best_index, cv2.CC_STAT_LEFT]),
            int(stats[best_index, cv2.CC_STAT_TOP]),
            int(stats[best_index, cv2.CC_STAT_WIDTH]),
            int(stats[best_index, cv2.CC_STAT_HEIGHT]),
            best_area,
        )

    @staticmethod
    def __coverage_ratio(*, bounds: Bounds, width: int, height: int) -> float:
        """
        Return the overlay coverage ratio against the screen area.
        """

        screen_area = max(1, width * height)
        return float(bounds.width * bounds.height) / float(screen_area)

    def __log_context(self, *, activity: str) -> Dict[str, Any]:
        """
        Return shared structured-logging context for overlay-detector entries.
        """

        return {
            "component": "adapter.perception.overlay.pixel",
            "workflow.id": self.__workflow_id,
            "activity": activity,
        }
