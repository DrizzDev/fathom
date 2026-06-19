from __future__ import annotations

import io
import unittest

from PIL import Image

from fathom.core.perception.orientation import CaptureOrientationResolver


class CaptureOrientationResolverTest(unittest.TestCase):
    """
    Pins the orientation alignment between device-reported dims and screenshot aspect.
    """

    @staticmethod
    def __png_bytes(*, width: int, height: int) -> bytes:
        """
        Encode a minimal PNG carrying the requested pixel dimensions.
        """

        buffer = io.BytesIO()
        Image.new("RGB", (width, height), "white").save(buffer, format="PNG")

        return buffer.getvalue()

    def test_landscape_image_with_portrait_report_swaps_dims(self) -> None:
        """
        Cooking-Craze-style mismatch: image is wider than tall, device reports portrait.
        """

        image = self.__png_bytes(width=2340, height=1080)

        corrected = CaptureOrientationResolver.resolve(
            image=image,
            reported_width=1080,
            reported_height=2340,
        )

        self.assertEqual(corrected, (2340, 1080))

    def test_portrait_image_with_landscape_report_swaps_dims(self) -> None:
        """
        Reverse mismatch: image is taller than wide, device reports landscape.
        """

        image = self.__png_bytes(width=1080, height=2340)

        corrected = CaptureOrientationResolver.resolve(
            image=image,
            reported_width=2340,
            reported_height=1080,
        )

        self.assertEqual(corrected, (1080, 2340))

    def test_portrait_image_with_portrait_report_keeps_dims(self) -> None:
        """
        Aligned portrait orientations require no correction.
        """

        image = self.__png_bytes(width=1080, height=2340)

        corrected = CaptureOrientationResolver.resolve(
            image=image,
            reported_width=1080,
            reported_height=2340,
        )

        self.assertEqual(corrected, (1080, 2340))

    def test_landscape_image_with_landscape_report_keeps_dims(self) -> None:
        """
        Aligned landscape orientations require no correction.
        """

        image = self.__png_bytes(width=2340, height=1080)

        corrected = CaptureOrientationResolver.resolve(
            image=image,
            reported_width=2340,
            reported_height=1080,
        )

        self.assertEqual(corrected, (2340, 1080))

    def test_retina_landscape_image_keeps_reported_logical(self) -> None:
        """
        iOS retina landscape: 3x pixel image is landscape and the reported logical is landscape too.
        """

        image = self.__png_bytes(width=2436, height=1125)

        corrected = CaptureOrientationResolver.resolve(
            image=image,
            reported_width=812,
            reported_height=375,
        )

        self.assertEqual(corrected, (812, 375))

    def test_square_image_returns_reported_dims_unchanged(self) -> None:
        """
        Square aspect is ambiguous; correction must not guess.
        """

        image = self.__png_bytes(width=512, height=512)

        corrected = CaptureOrientationResolver.resolve(
            image=image,
            reported_width=1080,
            reported_height=2340,
        )

        self.assertEqual(corrected, (1080, 2340))

    def test_empty_image_returns_reported_dims_unchanged(self) -> None:
        """
        No image bytes means no signal — caller's report is preserved.
        """

        corrected = CaptureOrientationResolver.resolve(
            image=b"",
            reported_width=1080,
            reported_height=2340,
        )

        self.assertEqual(corrected, (1080, 2340))

    def test_corrupt_image_returns_reported_dims_unchanged(self) -> None:
        """
        A malformed PNG body must not crash the capture pipeline.
        """

        corrected = CaptureOrientationResolver.resolve(
            reported_width=1080,
            reported_height=2340,
            image=b"not a png at all",
        )

        self.assertEqual(corrected, (1080, 2340))

    def test_zero_reported_dimension_returns_reported_unchanged(self) -> None:
        """
        Invalid reported dims short-circuit so the boundary stays fail-safe.
        """

        image = self.__png_bytes(width=2340, height=1080)

        corrected = CaptureOrientationResolver.resolve(
            image=image,
            reported_width=0,
            reported_height=2340,
        )

        self.assertEqual(corrected, (0, 2340))
