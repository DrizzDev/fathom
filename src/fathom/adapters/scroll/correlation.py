from __future__ import annotations

from io import BytesIO
from typing import Optional, Tuple

from PIL import Image

from fathom.constants.scroll import (
    DEFAULT_SCROLL_CORRELATION_STEP,
    DEFAULT_SCROLL_HIGH_CONFIDENCE,
    DEFAULT_SCROLL_LOW_CONFIDENCE,
    DEFAULT_SCROLL_MAXIMUM_TRANSLATION_RATIO,
    DEFAULT_SCROLL_MINIMUM_TRANSLATION,
    ScrollDirection,
    ScrollEvidenceSource,
    ScrollVerdictKind,
)
from fathom.interfaces.scroll import ScrollDetectPort
from fathom.schemas.actions import Bounds
from fathom.schemas.screens import ScreenCapture
from fathom.schemas.scroll import ScrollVerdict


class PhaseCorrelationScrollDetector(ScrollDetectPort):
    """
    Deterministic detector for scroll movement within a capture region.
    """

    __CONFIDENCE_MARGIN = 0.08
    __MAGNITUDE_MARGIN_RATIO = 1.20

    def __init__(
        self,
        *,
        high_confidence: float = DEFAULT_SCROLL_HIGH_CONFIDENCE,
        low_confidence: float = DEFAULT_SCROLL_LOW_CONFIDENCE,
        minimum_translation: int = DEFAULT_SCROLL_MINIMUM_TRANSLATION,
        maximum_translation_ratio: float = DEFAULT_SCROLL_MAXIMUM_TRANSLATION_RATIO,
        correlation_step: int = DEFAULT_SCROLL_CORRELATION_STEP,
    ) -> None:
        """
        Bind detector thresholds.
        """

        self.__high_confidence = high_confidence
        self.__low_confidence = low_confidence
        self.__minimum_translation = minimum_translation
        self.__maximum_translation_ratio = maximum_translation_ratio
        self.__correlation_step = correlation_step

    async def evaluate(
        self,
        *,
        before: ScreenCapture,
        after: ScreenCapture,
        region: Bounds,
        direction: ScrollDirection,
    ) -> ScrollVerdict:
        """
        Evaluate movement inside the requested capture region.
        """

        before_crop = self.__crop(capture=before, region=region)
        after_crop = self.__crop(capture=after, region=region)

        if before_crop is None or after_crop is None:
            return self.__verdict(
                kind=ScrollVerdictKind.AMBIGUOUS,
                confidence=0.0,
                distance=0,
                detail="crop_failed",
            )

        if before_crop.tobytes() == after_crop.tobytes():
            return self.__verdict(
                kind=ScrollVerdictKind.NO_PROGRESS,
                confidence=1.0,
                distance=0,
                detail="region_stable",
            )

        vertical_distance, vertical_score = self.__search(
            before=before_crop,
            after=after_crop,
            vertical=True,
        )
        horizontal_distance, horizontal_score = self.__search(
            before=before_crop,
            after=after_crop,
            vertical=False,
        )

        return self.__classify(
            direction=direction,
            region=region,
            vertical_distance=vertical_distance,
            vertical_score=vertical_score,
            horizontal_distance=horizontal_distance,
            horizontal_score=horizontal_score,
        )

    def __classify(
        self,
        *,
        direction: ScrollDirection,
        region: Bounds,
        vertical_distance: int,
        vertical_score: float,
        horizontal_distance: int,
        horizontal_score: float,
    ) -> ScrollVerdict:
        """
        Convert raw correlation scores into a typed verdict.
        """

        expected_vertical = direction in {ScrollDirection.UP, ScrollDirection.DOWN}
        primary_distance = vertical_distance if expected_vertical else horizontal_distance
        primary_score = vertical_score if expected_vertical else horizontal_score
        secondary_distance = horizontal_distance if expected_vertical else vertical_distance
        secondary_score = horizontal_score if expected_vertical else vertical_score
        axis_extent = region.height if expected_vertical else region.width
        maximum_translation = int(axis_extent * self.__maximum_translation_ratio)
        primary_magnitude = abs(primary_distance)
        secondary_magnitude = abs(secondary_distance)

        if (
            primary_score >= self.__high_confidence
            and primary_magnitude < self.__minimum_translation
        ):
            return self.__verdict(
                kind=ScrollVerdictKind.NO_PROGRESS,
                confidence=primary_score,
                distance=0,
                detail="region_stable",
            )

        if (
            primary_score >= self.__high_confidence
            and self.__matches_direction(direction=direction, distance=primary_distance)
            and self.__minimum_translation <= primary_magnitude <= maximum_translation
        ):
            return self.__verdict(
                kind=ScrollVerdictKind.PROGRESSED,
                confidence=primary_score,
                distance=primary_magnitude,
                detail="axis_progress_confirmed",
            )

        if self.__likely_axis_progress(
            direction=direction,
            primary_distance=primary_distance,
            primary_magnitude=primary_magnitude,
            primary_score=primary_score,
            secondary_magnitude=secondary_magnitude,
            secondary_score=secondary_score,
            maximum_translation=maximum_translation,
        ):
            return self.__verdict(
                kind=ScrollVerdictKind.PROGRESSED,
                confidence=primary_score,
                distance=primary_magnitude,
                detail="axis_progress_likely",
            )

        if (
            secondary_score >= self.__high_confidence
            and secondary_magnitude > primary_magnitude
            and secondary_magnitude >= self.__minimum_translation
        ):
            return self.__verdict(
                kind=ScrollVerdictKind.WRONG_AXIS,
                confidence=secondary_score,
                distance=secondary_magnitude,
                detail="movement_detected_on_other_axis",
            )

        if primary_score <= self.__low_confidence and secondary_score <= self.__low_confidence:
            return self.__verdict(
                kind=ScrollVerdictKind.AMBIGUOUS,
                confidence=float(max(primary_score, secondary_score)),
                distance=max(primary_magnitude, secondary_magnitude),
                detail="region_changed_without_stable_translation",
            )

        return self.__verdict(
            kind=ScrollVerdictKind.AMBIGUOUS,
            confidence=float(max(primary_score, secondary_score)),
            distance=max(primary_magnitude, secondary_magnitude),
            detail="translation_in_uncertain_band",
        )

    def __likely_axis_progress(
        self,
        *,
        direction: ScrollDirection,
        primary_distance: int,
        primary_magnitude: int,
        primary_score: float,
        secondary_magnitude: int,
        secondary_score: float,
        maximum_translation: int,
    ) -> bool:
        """
        Return whether the intended axis progressed even without high-confidence correlation.
        """

        if primary_score <= self.__low_confidence:
            return False

        if not self.__matches_direction(direction=direction, distance=primary_distance):
            return False

        if not (self.__minimum_translation <= primary_magnitude <= maximum_translation):
            return False

        score_leads_axis = primary_score >= (secondary_score + self.__CONFIDENCE_MARGIN)
        magnitude_leads_axis = primary_magnitude >= max(
            self.__minimum_translation,
            int(secondary_magnitude * self.__MAGNITUDE_MARGIN_RATIO),
        )
        return score_leads_axis or magnitude_leads_axis

    @staticmethod
    def __matches_direction(*, direction: ScrollDirection, distance: int) -> bool:
        """
        Return whether the observed movement sign matches the intended direction.
        """

        if direction in {ScrollDirection.UP, ScrollDirection.LEFT}:
            return distance < 0
        return distance > 0

    def __search(
        self,
        *,
        before: Image.Image,
        after: Image.Image,
        vertical: bool,
    ) -> Tuple[int, float]:
        """
        Search the strongest translation along one axis.
        """

        width, height = before.size
        extent = height if vertical else width
        best_distance = 0
        best_score = -1.0

        for distance in range(-extent + 1, extent, self.__correlation_step):
            score = self.__score(
                before=before,
                after=after,
                distance=distance,
                vertical=vertical,
            )
            if score > best_score or (score == best_score and abs(distance) < abs(best_distance)):
                best_score = score
                best_distance = distance

        return best_distance, max(0.0, best_score)

    def __score(
        self,
        *,
        before: Image.Image,
        after: Image.Image,
        distance: int,
        vertical: bool,
    ) -> float:
        """
        Score one candidate translation by average grayscale agreement.
        """

        before_gray = before.convert("L")
        after_gray = after.convert("L")
        width, height = before_gray.size

        if vertical:
            top_before = max(0, distance)
            top_after = max(0, -distance)
            overlap = height - abs(distance)
            if overlap <= 0:
                return 0.0
            before_crop = before_gray.crop((0, top_before, width, top_before + overlap))
            after_crop = after_gray.crop((0, top_after, width, top_after + overlap))
        else:
            left_before = max(0, distance)
            left_after = max(0, -distance)
            overlap = width - abs(distance)
            if overlap <= 0:
                return 0.0
            before_crop = before_gray.crop((left_before, 0, left_before + overlap, height))
            after_crop = after_gray.crop((left_after, 0, left_after + overlap, height))

        before_pixels = before_crop.load()
        after_pixels = after_crop.load()
        if before_pixels is None or after_pixels is None:
            return 0.0

        total = 0.0
        count = before_crop.size[0] * before_crop.size[1]
        if count == 0:
            return 0.0

        for x in range(before_crop.size[0]):
            for y in range(before_crop.size[1]):
                total += abs(before_pixels[x, y] - after_pixels[x, y])

        return float(1.0 - (total / (count * 255.0)))

    @staticmethod
    def __crop(*, capture: ScreenCapture, region: Bounds) -> Optional[Image.Image]:
        """
        Crop one capture to the requested bounds.
        """

        try:
            image = Image.open(BytesIO(capture.image))
        except Exception:
            return None

        left = max(0, region.x)
        top = max(0, region.y)
        right = min(image.width, region.x + region.width)
        bottom = min(image.height, region.y + region.height)
        if right <= left or bottom <= top:
            return None

        return image.crop((left, top, right, bottom))

    @staticmethod
    def __verdict(
        *,
        kind: ScrollVerdictKind,
        confidence: float,
        distance: int,
        detail: Optional[str],
    ) -> ScrollVerdict:
        """
        Build a detector verdict.
        """

        clamped = max(0.0, min(1.0, confidence))
        return ScrollVerdict(
            kind=kind,
            source=ScrollEvidenceSource.CORRELATION,
            confidence=clamped,
            distance=distance,
            detail=detail,
        )
