from __future__ import annotations

import unittest

import cv2
import numpy

from fathom.adapters.perception.overlay.pixel import PixelOverlayDetector
from fathom.schemas.budgets import PerceptionBudget
from fathom.schemas.screens import ScreenCapture


class PixelOverlayDetectorTest(unittest.IsolatedAsyncioTestCase):
    """
    Pins the OpenCV pixel-overlay scrim detector.

    The detector classifies a uniform dark region as a scrim when three
    thresholds are met: average intensity ≤ ``PIXEL_OVERLAY_MAX_INTENSITY``,
    area ≥ ``PIXEL_OVERLAY_MIN_AREA_RATIO`` × screen, and variance ≤
    ``PIXEL_OVERLAY_MAX_VARIANCE``. The tests synthesise grayscale PNGs
    that exercise each gate explicitly so a future threshold tweak fails
    the relevant pin rather than silently changing detection behaviour.
    """

    @staticmethod
    def __encode(image: numpy.ndarray) -> bytes:
        """
        Encode a grayscale numpy array as PNG bytes. Mirrors what a real
        device capture would deliver to the detector; centralised here so
        every synthetic fixture follows the same encoding path.
        """

        ok, buffer = cv2.imencode(".png", image)
        if not ok:
            raise RuntimeError("Failed to encode synthetic PNG fixture")
        return bytes(buffer.tobytes())

    @classmethod
    def __scrim_image(cls, *, intensity: int = 20) -> bytes:
        """
        Full-frame near-uniform dark canvas at intensity ``20`` — well
        below the ``MAX_INTENSITY`` floor and with near-zero variance.
        Exercises the happy-path detection.
        """

        canvas = numpy.full((400, 400), intensity, dtype=numpy.uint8)
        return cls.__encode(canvas)

    @classmethod
    def __bright_image(cls) -> bytes:
        """
        Uniformly bright canvas at intensity ``240`` — every pixel fails
        the intensity threshold, so the thresholded mask has no dim
        component and the detector returns ``None``.
        """

        canvas = numpy.full((400, 400), 240, dtype=numpy.uint8)
        return cls.__encode(canvas)

    @classmethod
    def __small_dark_region_image(cls) -> bytes:
        """
        Bright canvas with a 20x20 dark blob (≈0.25% area) — passes the
        intensity gate but fails the area-ratio gate, exercising the
        small-blob rejection path.
        """

        canvas = numpy.full((400, 400), 220, dtype=numpy.uint8)
        canvas[10:30, 10:30] = 20
        return cls.__encode(canvas)

    @staticmethod
    def __capture(*, image: bytes) -> ScreenCapture:
        """
        :class:`ScreenCapture` fixture wrapping the synthetic PNG bytes.
        Width and height match the canvas so the detector's pixel
        coordinates align with the image rectangle.
        """

        return ScreenCapture(
            width=400,
            height=400,
            activity="app",
            image=image,
            timestamp=0,
        )

    @staticmethod
    def __budget() -> PerceptionBudget:
        """
        :class:`PerceptionBudget` with ``local=2000`` ms — the detector's
        ``asyncio.wait_for`` runs the OpenCV pipeline on a worker thread
        and the budget grants it two seconds to finish.
        """

        return PerceptionBudget(ocr=0, local=2000, localization=0)

    async def test_uniform_dark_scrim_detected(self) -> None:
        """
        A full-frame uniform dark region clears all three gates and the
        detector returns non-empty :class:`Bounds`.
        """

        bounds = await PixelOverlayDetector().detect(
            capture=self.__capture(image=self.__scrim_image()),
            budget=self.__budget(),
        )

        self.assertIsNotNone(bounds)
        assert bounds is not None
        self.assertGreater(bounds.width * bounds.height, 0)

    async def test_bright_image_fails_intensity_gate(self) -> None:
        """
        A bright canvas fails the intensity threshold; the connected-
        components mask is empty and the detector returns ``None``.
        """

        bounds = await PixelOverlayDetector().detect(
            capture=self.__capture(image=self.__bright_image()),
            budget=self.__budget(),
        )

        self.assertIsNone(bounds)

    async def test_small_dark_region_fails_area_gate(self) -> None:
        """
        A dark blob below the minimum area ratio is rejected even
        though it passes the intensity gate. Without this guard the
        detector would surface every small icon as a scrim.
        """

        bounds = await PixelOverlayDetector().detect(
            capture=self.__capture(image=self.__small_dark_region_image()),
            budget=self.__budget(),
        )

        self.assertIsNone(bounds)

    async def test_empty_image_short_circuits(self) -> None:
        """
        An empty image payload must short-circuit before the OpenCV
        pipeline runs. The detector returns ``None`` without raising.
        """

        bounds = await PixelOverlayDetector().detect(
            capture=self.__capture(image=b""),
            budget=self.__budget(),
        )

        self.assertIsNone(bounds)
