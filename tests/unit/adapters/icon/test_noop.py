from __future__ import annotations

import unittest

from fathom.adapters.icon.noop import NoopIconDetector
from fathom.schemas.budgets import PerceptionBudget
from fathom.schemas.screens import ScreenCapture


class NoopIconDetectorTest(unittest.IsolatedAsyncioTestCase):
    """
    Pins the noop icon-detector contract. Used as the default
    ``IconDetectorPort`` until a real template registry ships, so the test
    asserts that the adapter never inspects pixels or calls a provider.
    """

    @staticmethod
    def __capture() -> ScreenCapture:
        """
        Minimal :class:`ScreenCapture` fixture. The noop adapter ignores the
        image bytes; any non-empty payload satisfies the schema.
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
        Permissive perception budget so a future budget-aware adapter would
        not interpret the fixture as a throttled run.
        """

        return PerceptionBudget(ocr=500, local=500, localization=500)

    async def test_detect_returns_empty_matches(self) -> None:
        """
        ``detect`` must yield an empty match tuple and zero duration,
        signalling that no template matching was performed.
        """

        result = await NoopIconDetector().detect(
            capture=self.__capture(),
            budget=self.__budget(),
        )

        self.assertEqual(result.matches, ())
        self.assertEqual(result.duration, 0)
