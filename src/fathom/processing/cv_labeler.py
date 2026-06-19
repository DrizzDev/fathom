from __future__ import annotations

from logging import getLogger
from typing import Iterable, List, Optional

from fathom.schemas.ui import LabeledElement, UIBounds

logger = getLogger(__name__)


class VisualControlLabeler:
    """
    Screenshot-based detector for visually rendered controls missing from platform accessibility hierarchies.

    This is intentionally conservative: it only emits sizeable, saturated rectangular controls that are not already
    covered by an existing hierarchy element. The output enters the normal manifest/annotation pipeline with ``source=cv`` provenance.
    """

    __MIN_WIDTH = 120
    __MIN_HEIGHT = 56
    __MIN_AREA = 8_000
    __MIN_FILL_RATIO = 0.35
    __MAX_WIDTH_RATIO = 0.85
    __MAX_HEIGHT_RATIO = 0.18
    __OVERLAP_IOU_THRESHOLD = 0.35

    @classmethod
    def detect(
        cls,
        *,
        image: bytes,
        scale_factor: float,
        existing_elements: Iterable[LabeledElement],
    ) -> List[LabeledElement]:
        """
        Return additional logical-coordinate elements for visual controls.

        Consumes in-memory screenshot bytes so the detector stays
        independent of any filesystem-staging artifact lifecycle.
        """

        if not image:
            logger.warning("[CVLabeler] empty screenshot bytes; skipping visual controls")
            return []

        try:
            import cv2
            import numpy as np
        except Exception as exception:
            logger.warning(
                "[CVLabeler] OpenCV unavailable; skipping visual controls: %s", exception
            )
            return []

        buffer = np.frombuffer(image, dtype=np.uint8)
        decoded = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if decoded is None:
            logger.warning("[CVLabeler] unable to decode screenshot bytes")
            return []

        height, width = decoded.shape[:2]
        hsv = cv2.cvtColor(decoded, cv2.COLOR_BGR2HSV)

        # Saturated + bright regions capture filled CTA/buttons while ignoring dark overlay masks and white text.
        mask = ((hsv[:, :, 1] > 80) & (hsv[:, :, 2] > 120)).astype(np.uint8) * 255

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        existing_pixel_bounds = [
            cls.__scale_bounds(bounds=element.bounds, scale_factor=scale_factor)
            for element in existing_elements
        ]

        detected: List[LabeledElement] = []

        for index in range(1, count):
            x, y, candidate_width, candidate_height, area = (int(value) for value in stats[index])
            candidate = UIBounds(
                x1=x,
                y1=y,
                x2=x + candidate_width,
                y2=y + candidate_height,
            )

            if not cls.__is_viable_control(
                area=area,
                bounds=candidate,
                screen_width=width,
                screen_height=height,
            ):
                continue

            if cls.__overlaps_existing(bounds=candidate, existing_bounds=existing_pixel_bounds):
                continue

            logical = cls.__unscale_bounds(bounds=candidate, scale_factor=scale_factor)
            detected.append(
                LabeledElement(
                    label="",
                    color="",
                    bounds=logical,
                    attributes={
                        "source": "cv",
                        "confidence": "0.85",
                        "type": "VisualControl",
                        "class": "VisualControl",
                    },
                )
            )

        if detected:
            logger.info("[CVLabeler] added %d visual control label(s)", len(detected))

        return detected

    @classmethod
    def __is_viable_control(
        cls,
        *,
        area: int,
        bounds: UIBounds,
        screen_width: int,
        screen_height: int,
    ) -> bool:
        """ """

        candidate_width = bounds.width
        candidate_height = bounds.height

        if candidate_width < cls.__MIN_WIDTH or candidate_height < cls.__MIN_HEIGHT:
            return False

        if candidate_width > screen_width * cls.__MAX_WIDTH_RATIO:
            return False

        if candidate_height > screen_height * cls.__MAX_HEIGHT_RATIO:
            return False

        if area < cls.__MIN_AREA:
            return False

        return area / max(1.0, candidate_width * candidate_height) >= cls.__MIN_FILL_RATIO

    @classmethod
    def __overlaps_existing(
        cls,
        *,
        bounds: UIBounds,
        existing_bounds: Iterable[UIBounds],
    ) -> bool:
        """ """

        return any(
            cls.__iou(first=bounds, second=existing) >= cls.__OVERLAP_IOU_THRESHOLD
            for existing in existing_bounds
        )

    @staticmethod
    def __scale_bounds(*, bounds: UIBounds, scale_factor: float) -> UIBounds:
        """ """

        return UIBounds(
            x1=bounds.x1 * scale_factor,
            y1=bounds.y1 * scale_factor,
            x2=bounds.x2 * scale_factor,
            y2=bounds.y2 * scale_factor,
        )

    @staticmethod
    def __unscale_bounds(*, bounds: UIBounds, scale_factor: float) -> UIBounds:
        """ """

        safe_scale = scale_factor if scale_factor > 0 else 1.0

        return UIBounds(
            x1=bounds.x1 / safe_scale,
            y1=bounds.y1 / safe_scale,
            x2=bounds.x2 / safe_scale,
            y2=bounds.y2 / safe_scale,
        )

    @staticmethod
    def __iou(*, first: UIBounds, second: Optional[UIBounds]) -> float:
        """ """

        if second is None:
            return 0.0

        top = max(first.y1, second.y1)
        left = max(first.x1, second.x1)
        right = min(first.x2, second.x2)
        bottom = min(first.y2, second.y2)

        if right <= left or bottom <= top:
            return 0.0

        first_area = first.width * first.height
        second_area = second.width * second.height
        intersection = (right - left) * (bottom - top)

        union = first_area + second_area - intersection
        if union <= 0:
            return 0.0

        return float(intersection / union)
