from __future__ import annotations

import time
from logging import getLogger
from typing import List, Optional, Tuple, TypeAlias, cast

import cv2
import numpy
from numpy.typing import NDArray

from fathom.constants.screen import (
    DILATION_ITERATIONS,
    DILATION_KERNEL_SIZE,
    MAX_VISUAL_HASH_DISTANCE,
    MIN_CHANGED_REGION_AREA_PX,
    NAVIGATION_BAR_HEIGHT_PX,
    PIXEL_CHANGE_THRESHOLD,
    SSIM_GAUSSIAN_KERNEL_SIZE,
    SSIM_GAUSSIAN_SIGMA,
    SSIM_K1,
    SSIM_K2,
    STATUS_BAR_HEIGHT_PX,
    ZERO_HASH,
)
from fathom.schemas.screens import (
    ScreenCapture,
    ScreenChangeRegion,
    ScreenDiff,
    ScreenScrollTranslation,
    ScreenState,
    StructuralComparisonSignals,
)

logger = getLogger(__name__)
ImageMatrix: TypeAlias = NDArray[numpy.generic]


class ScreenComparator:
    """
    Compare consecutive captures using structural and visual signals.
    """

    def compare(
        self,
        *,
        after: ScreenCapture,
        before: ScreenCapture,
        after_state: Optional[ScreenState] = None,
        before_state: Optional[ScreenState] = None,
    ) -> ScreenDiff:
        """
        Compare two captures and return a rich diff object.
        """

        compare_start = time.time()

        structural_signals = self.__resolve_structural_signals(
            after_state=after_state,
            before_state=before_state,
        )

        ssim_start = time.time()
        ssim_score = self.__compute_ssim(after=after.image, before=before.image)
        logger.info("[ScreenDiff] SSIM completed in %.2fs", time.time() - ssim_start)

        pixel_start = time.time()
        pixel_diff = self.__compute_content_pixel_diff_ratio(after=after.image, before=before.image)
        logger.info("[ScreenDiff] Pixel diff completed in %.2fs", time.time() - pixel_start)

        regions_start = time.time()
        regions = self.__compute_changed_regions(after=after.image, before=before.image)
        logger.info("[ScreenDiff] Changed regions completed in %.2fs", time.time() - regions_start)

        scroll_start = time.time()
        scroll = self.__compute_scroll_translation(after=after.image, before=before.image)
        logger.info(
            "[ScreenDiff] Scroll translation completed in %.2fs", time.time() - scroll_start
        )

        logger.info("[ScreenDiff] Total compare completed in %.2fs", time.time() - compare_start)

        return ScreenDiff(
            ssim_score=ssim_score,
            changed_regions=regions,
            scroll_translation=scroll,
            content_pixel_diff_ratio=pixel_diff,
            phash_distance=structural_signals.phash_distance,
            activity_changed=before.activity != after.activity,
            xml_hash_changed=structural_signals.xml_hash_changed,
            interaction_hash_changed=structural_signals.interaction_hash_changed,
        )

    def __resolve_structural_signals(
        self,
        *,
        after_state: Optional[ScreenState],
        before_state: Optional[ScreenState],
    ) -> StructuralComparisonSignals:
        """
        Resolve structural change signals from the available screen states.
        """

        if before_state is None or after_state is None:
            return StructuralComparisonSignals(
                xml_hash_changed=False,
                interaction_hash_changed=False,
                phash_distance=MAX_VISUAL_HASH_DISTANCE,
            )

        phash_distance = ScreenState.hamming_distance(
            left_hash=before_state.visual_hash,
            right_hash=after_state.visual_hash,
        )
        xml_hash_changed = self.__did_hash_change(
            after_hash=after_state.xml_hash,
            before_hash=before_state.xml_hash,
        )
        interaction_hash_changed = self.__did_hash_change(
            after_hash=after_state.interaction_hash,
            before_hash=before_state.interaction_hash,
        )

        return StructuralComparisonSignals(
            phash_distance=phash_distance,
            xml_hash_changed=xml_hash_changed,
            interaction_hash_changed=interaction_hash_changed,
        )

    def __did_hash_change(
        self,
        *,
        after_hash: Optional[str],
        before_hash: Optional[str],
    ) -> bool:
        """
        Return whether two non-sentinel hashes differ.
        """

        if not before_hash or not after_hash:
            return False

        if before_hash == ZERO_HASH or after_hash == ZERO_HASH:
            return False

        return before_hash != after_hash

    def __decode_grayscale(self, *, image_data: bytes) -> Optional[ImageMatrix]:
        """
        Decode image bytes into a grayscale image array.
        """

        try:
            byte_array = numpy.frombuffer(image_data, numpy.uint8)
            decoded_image = cv2.imdecode(byte_array, cv2.IMREAD_GRAYSCALE)

            if decoded_image is None:
                return None

            return cast("ImageMatrix", decoded_image)
        except Exception as exception:
            logger.debug("Image decode failed: %s", exception)
            return None

    @staticmethod
    def __downscale_for_ssim(*, image: ImageMatrix, max_width: int = 540) -> ImageMatrix:
        """
        Downscale a grayscale image for SSIM computation.

        SSIM measures structural similarity which is preserved at lower resolutions.
        Downscaling from 1080px to 540px reduces pixel count by 4x and compute by ~4-8x.
        """

        height, width = image.shape[:2]

        if width <= max_width:
            return image

        scale = max_width / width
        new_width = max_width
        new_height = int(height * scale)

        return cast("ImageMatrix", cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA))

    def __get_content_bounds(self, *, image_height: int) -> Tuple[int, int]:
        """
        Return the vertical crop bounds for the meaningful content area.
        """

        top = min(STATUS_BAR_HEIGHT_PX, max(image_height - 1, 0))
        bottom = max(top + 1, image_height - NAVIGATION_BAR_HEIGHT_PX)
        bottom = min(bottom, image_height)

        return (top, bottom)

    def __get_content_region(self, *, image: ImageMatrix) -> ImageMatrix:
        """
        Exclude status and navigation bars from content-sensitive comparisons.
        """

        top, bottom = self.__get_content_bounds(image_height=image.shape[0])
        return image[top:bottom, :]

    def __compute_ssim(self, *, before: bytes, after: bytes) -> Optional[float]:
        """
        Compute SSIM over the content region.

        Images are downscaled to max 540px width before SSIM computation.
        SSIM is a structural metric designed to work at any resolution —
        downscaling preserves accuracy while reducing compute by 4-16x.
        """

        try:
            image_before = self.__decode_grayscale(image_data=before)
            image_after = self.__decode_grayscale(image_data=after)

            if image_before is None or image_after is None:
                return None

            if image_before.shape != image_after.shape:
                return 0.0

            image_before = self.__downscale_for_ssim(image=image_before)
            image_after = self.__downscale_for_ssim(image=image_after)

            return self.__compute_ssim_map_mean(
                after_image=self.__get_content_region(image=image_after),
                before_image=self.__get_content_region(image=image_before),
            )
        except Exception as exception:
            logger.debug("SSIM failed: %s", exception)
            return None

    def __compute_ssim_map_mean(
        self,
        *,
        after_image: ImageMatrix,
        before_image: ImageMatrix,
    ) -> float:
        """
        Compute the mean SSIM value using only OpenCV and NumPy.
        """

        sigma = SSIM_GAUSSIAN_SIGMA
        kernel_size = SSIM_GAUSSIAN_KERNEL_SIZE

        contrast_stabilizer = (SSIM_K2 * 255.0) ** 2
        luminance_stabilizer = (SSIM_K1 * 255.0) ** 2

        after_float = after_image.astype(numpy.float32)
        before_float = before_image.astype(numpy.float32)

        gaussian_column = cv2.getGaussianKernel(kernel_size, sigma)
        gaussian_kernel = numpy.outer(gaussian_column, gaussian_column.transpose())

        padding = kernel_size // 2

        before_mean = cv2.filter2D(before_float, -1, gaussian_kernel)[
            padding:-padding, padding:-padding
        ]
        after_mean = cv2.filter2D(after_float, -1, gaussian_kernel)[
            padding:-padding, padding:-padding
        ]

        before_mean_sq = before_mean * before_mean
        after_mean_sq = after_mean * after_mean
        mean_product = before_mean * after_mean

        before_variance = (
            cv2.filter2D(before_float * before_float, -1, gaussian_kernel)[
                padding:-padding, padding:-padding
            ]
            - before_mean_sq
        )
        after_variance = (
            cv2.filter2D(after_float * after_float, -1, gaussian_kernel)[
                padding:-padding, padding:-padding
            ]
            - after_mean_sq
        )
        covariance = (
            cv2.filter2D(before_float * after_float, -1, gaussian_kernel)[
                padding:-padding, padding:-padding
            ]
            - mean_product
        )

        numerator = (2.0 * mean_product + luminance_stabilizer) * (
            2.0 * covariance + contrast_stabilizer
        )
        denominator = (before_mean_sq + after_mean_sq + luminance_stabilizer) * (
            before_variance + after_variance + contrast_stabilizer
        )
        safe_denominator = numpy.where(denominator == 0, 1.0, denominator)
        ssim_map = numpy.divide(
            numerator,
            safe_denominator,
            out=numpy.ones_like(numerator, dtype=numpy.float64),
            where=denominator != 0,
        )
        return float(numpy.mean(ssim_map))

    def __compute_content_pixel_diff_ratio(self, *, before: bytes, after: bytes) -> Optional[float]:
        """
        Compute the ratio of changed pixels in the content region.
        """

        try:
            image_before = self.__decode_grayscale(image_data=before)
            image_after = self.__decode_grayscale(image_data=after)

            if image_before is None or image_after is None:
                return None

            if image_before.shape != image_after.shape:
                return 1.0

            before_region = self.__get_content_region(image=image_before)
            after_region = self.__get_content_region(image=image_after)
            diff = numpy.abs(before_region.astype(numpy.int16) - after_region.astype(numpy.int16))

            total_pixels = int(before_region.size)

            if total_pixels == 0:
                return None

            changed_pixels = int(numpy.sum(diff > PIXEL_CHANGE_THRESHOLD))
            return float(changed_pixels) / float(total_pixels)
        except Exception as exception:
            logger.debug("Pixel diff failed: %s", exception)
            return None

    def __compute_changed_regions(self, *, before: bytes, after: bytes) -> List[ScreenChangeRegion]:
        """
        Detect changed regions as bounding boxes in full-screen coordinates.
        """

        try:
            image_before = self.__decode_grayscale(image_data=before)
            image_after = self.__decode_grayscale(image_data=after)

            if image_before is None or image_after is None:
                return []

            if image_before.shape != image_after.shape:
                return []

            diff = cv2.absdiff(image_before, image_after)
            _, binary = cv2.threshold(diff, PIXEL_CHANGE_THRESHOLD, 255, cv2.THRESH_BINARY)

            kernel = numpy.ones((DILATION_KERNEL_SIZE, DILATION_KERNEL_SIZE), numpy.uint8)
            dilated = cv2.dilate(binary, kernel, iterations=DILATION_ITERATIONS)

            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            regions: List[ScreenChangeRegion] = []
            _, content_bottom = self.__get_content_bounds(image_height=image_before.shape[0])

            for contour in contours:
                if cv2.contourArea(contour) < MIN_CHANGED_REGION_AREA_PX:
                    continue

                x, y, width, height = cv2.boundingRect(contour)
                if y + height <= STATUS_BAR_HEIGHT_PX:
                    continue

                if y >= content_bottom:
                    continue

                regions.append(
                    ScreenChangeRegion(
                        x=int(x),
                        y=int(y),
                        width=int(width),
                        height=int(height),
                    )
                )

            return regions
        except Exception as exception:
            logger.debug("Changed region detection failed: %s", exception)
            return []

    def __compute_scroll_translation(
        self, *, before: bytes, after: bytes
    ) -> Optional[ScreenScrollTranslation]:
        """
        Estimate frame translation using phase correlation.
        """

        try:
            image_before = self.__decode_grayscale(image_data=before)
            image_after = self.__decode_grayscale(image_data=after)

            if image_before is None or image_after is None:
                return None

            if image_before.shape != image_after.shape:
                return None

            before_region = self.__get_content_region(image=image_before)
            after_region = self.__get_content_region(image=image_after)

            shift, _ = cv2.phaseCorrelate(
                before_region.astype(numpy.float32),
                after_region.astype(numpy.float32),
            )

            return ScreenScrollTranslation(
                dx=float(shift[0]),
                dy=float(shift[1]),
            )
        except Exception as exception:
            logger.debug("Phase correlation failed: %s", exception)
            return None
