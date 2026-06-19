from __future__ import annotations

import unittest

from fathom.adapters.ocr.noop import NoopOcr
from fathom.schemas.budgets import PerceptionBudget
from fathom.schemas.screens import ScreenCapture


class NoopOcrTest(unittest.IsolatedAsyncioTestCase):
    """
    Pins the noop OCR adapter contract: an inert ``OcrPort`` implementation
    used whenever the runtime is configured without Document AI credentials.
    The test verifies that ``extract`` returns an empty result without
    reaching any external provider.
    """

    @staticmethod
    def __capture() -> ScreenCapture:
        """
        Minimal :class:`ScreenCapture` fixture. The noop adapter never reads
        the image bytes, so any non-empty payload is acceptable.
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
        Permissive perception budget. Values are deliberately generous so a
        future budget-aware adapter would not see this fixture as throttled.
        """

        return PerceptionBudget(ocr=500, local=500, localization=500)

    async def test_extract_returns_empty_tokens_and_zero_duration(self) -> None:
        """
        ``extract`` must yield an empty token tuple and report zero
        duration, signalling to consumers that no provider was consulted.
        """

        result = await NoopOcr().extract(capture=self.__capture(), budget=self.__budget())

        self.assertEqual(result.tokens, ())
        self.assertEqual(result.duration, 0)
