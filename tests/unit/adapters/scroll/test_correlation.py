from __future__ import annotations

import io
import unittest

from PIL import Image, ImageDraw

from fathom.adapters.scroll.correlation import PhaseCorrelationScrollDetector
from fathom.constants.scroll import ScrollDirection, ScrollVerdictKind
from fathom.schemas.actions import Bounds, CoordinateSystem
from fathom.schemas.screens import ScreenCapture


class PhaseCorrelationScrollDetectorTest(unittest.IsolatedAsyncioTestCase):
    """
    Covers deterministic scroll verdicts on controlled image pairs.
    """

    async def test_does_not_report_no_progress_for_clear_vertical_translation(self) -> None:
        """
        A clear translated feed should not collapse to no progress.
        """

        detector = PhaseCorrelationScrollDetector(
            minimum_translation=4,
            correlation_step=4,
            high_confidence=0.50,
        )
        before = self.__capture(offset=0)
        after = self.__capture(offset=80)

        verdict = await detector.evaluate(
            before=before,
            after=after,
            region=Bounds(
                x=0,
                y=0,
                width=400,
                height=600,
                coordinate_system=CoordinateSystem.DEVICE_PIXEL,
            ),
            direction=ScrollDirection.UP,
        )

        self.assertNotEqual(verdict.kind, ScrollVerdictKind.NO_PROGRESS)
        self.assertGreaterEqual(verdict.distance, 4)

    async def test_reports_no_progress_for_identical_frames(self) -> None:
        """
        Classify identical crops as no progress.
        """

        detector = PhaseCorrelationScrollDetector(
            minimum_translation=20,
            correlation_step=4,
            high_confidence=0.80,
        )
        before = self.__capture(offset=0)
        after = self.__capture(offset=0)

        verdict = await detector.evaluate(
            before=before,
            after=after,
            region=Bounds(
                x=0,
                y=0,
                width=400,
                height=600,
                coordinate_system=CoordinateSystem.DEVICE_PIXEL,
            ),
            direction=ScrollDirection.UP,
        )

        self.assertEqual(verdict.kind, ScrollVerdictKind.NO_PROGRESS)

    async def test_low_confidence_directional_signal_is_promoted_to_progress(self) -> None:
        """
        A directional low-confidence signal should not fall into the uncertain band forever.
        """

        detector = PhaseCorrelationScrollDetector(
            minimum_translation=4,
            correlation_step=4,
            high_confidence=0.98,
            low_confidence=0.10,
        )
        verdict = detector._PhaseCorrelationScrollDetector__classify(  # noqa: SLF001
            direction=ScrollDirection.UP,
            region=Bounds(
                x=0,
                y=0,
                width=400,
                height=600,
                coordinate_system=CoordinateSystem.DEVICE_PIXEL,
            ),
            vertical_distance=-96,
            vertical_score=0.61,
            horizontal_distance=24,
            horizontal_score=0.47,
        )

        self.assertEqual(verdict.kind, ScrollVerdictKind.PROGRESSED)
        self.assertEqual(verdict.detail, "axis_progress_likely")

    async def test_high_confidence_other_axis_still_reports_wrong_axis(self) -> None:
        """
        Strong perpendicular motion must still win over the likely-progress fallback.
        """

        detector = PhaseCorrelationScrollDetector(
            minimum_translation=4,
            correlation_step=4,
            high_confidence=0.90,
            low_confidence=0.10,
        )
        verdict = detector._PhaseCorrelationScrollDetector__classify(  # noqa: SLF001
            direction=ScrollDirection.UP,
            region=Bounds(
                x=0,
                y=0,
                width=400,
                height=600,
                coordinate_system=CoordinateSystem.DEVICE_PIXEL,
            ),
            vertical_distance=-24,
            vertical_score=0.42,
            horizontal_distance=120,
            horizontal_score=0.95,
        )

        self.assertEqual(verdict.kind, ScrollVerdictKind.WRONG_AXIS)

    @staticmethod
    def __capture(*, offset: int) -> ScreenCapture:
        """
        Build a synthetic feed crop from a taller canvas.
        """

        canvas = Image.new("RGB", (400, 1000), color="white")
        draw = ImageDraw.Draw(canvas)
        colors = [
            "#d32f2f",
            "#1976d2",
            "#388e3c",
            "#f57c00",
            "#7b1fa2",
            "#455a64",
        ]
        for index in range(14):
            top = index * 70
            color = colors[index % len(colors)]
            draw.rectangle((30, top, 370, top + 54), fill=color, outline="black", width=2)
        image = canvas.crop((0, offset, 400, offset + 600))

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return ScreenCapture(
            width=400,
            height=600,
            activity="synthetic",
            image=buffer.getvalue(),
            timestamp=0,
        )
