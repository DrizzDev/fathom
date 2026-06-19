from __future__ import annotations

import unittest

from fathom.adapters.icon.template import TemplateIconDetector
from fathom.schemas.budgets import PerceptionBudget
from fathom.schemas.icon import IconKind, IconTemplate
from fathom.schemas.screens import ScreenCapture


class TemplateIconDetectorTest(unittest.IsolatedAsyncioTestCase):
    """
    Pins the OpenCV template icon-detector edge paths.

    The detector compiles every supplied :class:`IconTemplate` at
    construction and matches against grayscale screen captures at call
    time. This suite covers the zero-template, empty-image, and undecodable-
    template fast paths so a future regression cannot silently raise on
    malformed input.
    """

    @staticmethod
    def __capture(*, image: bytes = b"PNG") -> ScreenCapture:
        """
        :class:`ScreenCapture` fixture with overridable image bytes.

        Most tests pass an undecodable placeholder because they exercise
        the empty-registry or empty-image fast paths that short-circuit
        before any OpenCV decode is attempted.
        """

        return ScreenCapture(
            width=1000,
            height=2000,
            activity="app",
            image=image,
            timestamp=0,
        )

    @staticmethod
    def __budget() -> PerceptionBudget:
        """
        Permissive perception budget — the detector's ``asyncio.wait_for``
        timeout is local/1000 seconds, so the fixture grants the worker
        thread half a second to finish.
        """

        return PerceptionBudget(ocr=500, local=500, localization=500)

    async def test_empty_registry_short_circuits(self) -> None:
        """
        With no compiled templates, the detector must skip decoding the
        screen capture entirely and return an empty result.
        """

        result = await TemplateIconDetector().detect(
            capture=self.__capture(),
            budget=self.__budget(),
        )

        self.assertEqual(result.matches, ())
        self.assertEqual(result.duration, 0)

    async def test_empty_image_returns_no_matches(self) -> None:
        """
        An empty image payload must yield no matches even when the
        template registry is non-empty.
        """

        detector = TemplateIconDetector(
            templates=(IconTemplate(image=b"PNG", kind=IconKind.HEART),),
        )

        result = await detector.detect(capture=self.__capture(image=b""), budget=self.__budget())

        self.assertEqual(result.matches, ())

    async def test_undecodable_template_bytes_dropped_silently(self) -> None:
        """
        Templates whose bytes cv2.imdecode rejects must be skipped at
        construction without raising. The detector then has nothing to
        match against and returns an empty result.
        """

        detector = TemplateIconDetector(
            templates=(IconTemplate(image=b"not-a-real-png", kind=IconKind.HEART),),
        )

        result = await detector.detect(
            capture=self.__capture(image=b"also-not-a-real-png"),
            budget=self.__budget(),
        )

        self.assertEqual(result.matches, ())
