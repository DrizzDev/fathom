from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any, List

from fathom.constants.defect import DefectKind, DefectSeverity, DefectSignal, DefectSource
from fathom.core.defect.vision import VisionDefectDetector
from fathom.interfaces.llm import LLMPort
from fathom.schemas.defect import ScreenSnapshot
from fathom.schemas.results import GenerateResult


class _StubLLM(LLMPort):
    """
    LLM stub returning a preset result and counting generate calls.
    """

    def __init__(self, result: GenerateResult) -> None:
        self.__result = result
        self.calls = 0

    @property
    def model_name(self) -> str:
        return "stub"

    async def generate(self, **kwargs: Any) -> GenerateResult:
        self.calls += 1
        return self.__result

    async def cleanup(self) -> None:
        return None


def _tool_call(defects: List[dict]) -> SimpleNamespace:
    """
    Builds a detect_defects tool call carrying the given findings.
    """

    return SimpleNamespace(name="detect_defects", args={"defects": defects})


class VisionDefectDetectorTest(unittest.IsolatedAsyncioTestCase):
    """
    Verifies parsing of detect_defects tool calls into anchored defects.
    """

    @staticmethod
    def __snapshot() -> ScreenSnapshot:
        return ScreenSnapshot(screen="home", activity="com.app/.Home", screenshot=b"PNG")

    async def test_no_screenshot_skips_the_model(self) -> None:
        """
        With no screenshot the detector returns nothing and never calls the model.
        """

        llm = _StubLLM(GenerateResult())
        detector = VisionDefectDetector(llm=llm)

        result = await detector.inspect_screen(snapshot=ScreenSnapshot(screen="home"))

        self.assertEqual(result, [])
        self.assertEqual(llm.calls, 0)

    async def test_parses_findings_into_defects(self) -> None:
        """
        Findings become anchored defects with explicit or defaulted severity.
        """

        call = _tool_call(
            [
                {
                    "signal": "overlap_clipping",
                    "severity": "major",
                    "summary": "Title overlaps the cart icon",
                    "bounds": {"x": 100, "y": 50, "width": 200, "height": 30},
                },
                {"signal": "lorem_ipsum", "summary": "Body copy is lorem ipsum"},
            ]
        )
        detector = VisionDefectDetector(llm=_StubLLM(GenerateResult(tool_calls=[call])))

        defects = await detector.inspect_screen(snapshot=self.__snapshot())

        self.assertEqual(
            [defect.signal for defect in defects],
            [DefectSignal.OVERLAP_CLIPPING, DefectSignal.LOREM_IPSUM],
        )
        self.assertEqual(defects[0].severity, DefectSeverity.MAJOR)
        self.assertEqual(defects[0].kind, DefectKind.UI)
        self.assertEqual(defects[0].source, DefectSource.POST_RUN)
        self.assertEqual(defects[0].evidence.screen, "home")
        self.assertIsNotNone(defects[0].evidence.bounds)
        self.assertEqual(defects[1].severity, DefectSignal.LOREM_IPSUM.default_severity)

    async def test_skips_invalid_signal_and_empty_summary(self) -> None:
        """
        Unknown signals and empty summaries are dropped, not raised.
        """

        call = _tool_call(
            [
                {"signal": "not_a_signal", "summary": "ignored"},
                {"signal": "contrast", "summary": ""},
                {"signal": "contrast", "summary": "Caption text is too faint"},
            ]
        )
        detector = VisionDefectDetector(llm=_StubLLM(GenerateResult(tool_calls=[call])))

        defects = await detector.inspect_screen(snapshot=self.__snapshot())

        self.assertEqual([defect.signal for defect in defects], [DefectSignal.CONTRAST])

    async def test_no_tool_call_returns_empty(self) -> None:
        """
        A response without the tool call yields no defects.
        """

        detector = VisionDefectDetector(llm=_StubLLM(GenerateResult(content="ok")))

        self.assertEqual(await detector.inspect_screen(snapshot=self.__snapshot()), [])


if __name__ == "__main__":
    unittest.main()
