from __future__ import annotations

import unittest

from fathom.adapters.perception.overlay.noop import NoopOverlayDetector
from fathom.schemas.budgets import PerceptionBudget
from fathom.schemas.screens import ScreenCapture


class NoopOverlayDetectorTest(unittest.IsolatedAsyncioTestCase):
    """
    Pins the noop overlay-detector contract. The runtime falls back to this
    adapter when perception is disabled; the test asserts that no pixel-
    level work happens and the call returns ``None`` cleanly.
    """

    @staticmethod
    def __capture() -> ScreenCapture:
        """
        Minimal :class:`ScreenCapture` fixture. The noop adapter never
        decodes the image bytes; any non-empty payload satisfies the schema.
        """

        return ScreenCapture(
            width=100,
            height=200,
            activity="app",
            image=b"PNG",
            timestamp=0,
        )

    @staticmethod
    def __budget() -> PerceptionBudget:
        """
        Permissive perception budget. Values are deliberately generous so
        the fixture is never interpreted as throttled.
        """

        return PerceptionBudget(ocr=500, local=500, localization=500)

    async def test_detect_returns_none(self) -> None:
        """
        ``detect`` must return ``None`` to indicate no overlay was found
        without consulting any pixel-level evidence.
        """

        result = await NoopOverlayDetector().detect(
            capture=self.__capture(),
            budget=self.__budget(),
        )

        self.assertIsNone(result)
