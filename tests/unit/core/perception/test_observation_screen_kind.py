from __future__ import annotations

import unittest
from pathlib import Path
from typing import Tuple
from unittest.mock import AsyncMock, Mock

from fathom.constants.observation import KeyboardVisibility
from fathom.core.perception.observation import ScreenObservationService
from fathom.schemas.budgets import PerceptionBudget
from fathom.schemas.ocr import OcrResult
from fathom.schemas.screens import ScreenCapture, ScreenHashBundle
from fathom.schemas.ui import LabeledElement

_FRAMES = Path(__file__).resolve().parents[3] / "fixtures" / "perception" / "frames"
_IMAGE = _FRAMES / "home.png"


class ScreenObservationServiceScreenKindTest(unittest.IsolatedAsyncioTestCase):
    """
    Pin the OCR-skip and OCR-run paths in :class:`ScreenObservationService`
    against synthetic capture fixtures whose ``xml_content`` deterministically
    resolves to the three :class:`ScreenKind` values.
    """

    @staticmethod
    def __capture(*, xml_content: str | None) -> ScreenCapture:
        """
        Build a capture fixture with the supplied hierarchy XML.
        """

        return ScreenCapture(
            width=2340,
            height=1080,
            timestamp=0,
            activity="app",
            xml_content=xml_content,
            image=_IMAGE.read_bytes(),
        )

    @staticmethod
    def __hashes() -> ScreenHashBundle:
        """
        Deterministic hash bundle for the screen-kind tests.
        """

        return ScreenHashBundle(
            xml_hash="a" * 16,
            visual_hash="0" * 16,
            interaction_hash="b" * 16,
        )

    @staticmethod
    def __budget() -> PerceptionBudget:
        """
        Generous budget so OCR would otherwise run.
        """

        return PerceptionBudget(ocr=30_000, local=5_000, localization=60_000)

    @staticmethod
    def __manifest() -> Tuple[LabeledElement, ...]:
        """
        Empty manifest forces the OCR-trigger heuristic to vote yes.
        """

        return ()

    @staticmethod
    def __ocr_mock() -> Mock:
        """
        Spy OCR port returning an empty result.
        """

        ocr = Mock()
        ocr.extract = AsyncMock(return_value=OcrResult(duration=0, raw_response=None, tokens=()))

        return ocr

    async def test_native_frame_runs_ocr(self) -> None:
        """
        A frame whose XML has no WebView / SurfaceView must run OCR normally.
        """

        ocr = self.__ocr_mock()
        await ScreenObservationService(ocr=ocr).observe(
            step_number=0,
            session_id="run-test",
            hashes=self.__hashes(),
            budget=self.__budget(),
            manifest=self.__manifest(),
            capture=self.__capture(
                xml_content='<root><android.widget.Button bounds="[0,0][50,50]"/></root>'
            ),
        )

        ocr.extract.assert_awaited_once()

    async def test_webview_frame_forces_ocr(self) -> None:
        """
        WebView hierarchies are opaque (one node, no clickable children); OCR
        is the only signal that can populate the manifest, so it must run
        regardless of the text-coverage heuristic.
        """

        ocr = self.__ocr_mock()
        await ScreenObservationService(ocr=ocr).observe(
            step_number=0,
            session_id="run-test",
            hashes=self.__hashes(),
            budget=self.__budget(),
            manifest=self.__manifest(),
            capture=self.__capture(
                xml_content=('<root><android.webkit.WebView bounds="[0,0][2340,1080]"/></root>')
            ),
        )

        ocr.extract.assert_awaited_once()

    async def test_game_surface_frame_forces_ocr(self) -> None:
        """
        OpenGL / SurfaceView roots expose no element tree; OCR must run to
        produce any actionable manifest at all.
        """

        ocr = self.__ocr_mock()
        await ScreenObservationService(ocr=ocr).observe(
            step_number=0,
            session_id="run-test",
            hashes=self.__hashes(),
            budget=self.__budget(),
            manifest=self.__manifest(),
            capture=self.__capture(
                xml_content=('<root><android.view.SurfaceView bounds="[0,0][2340,1080]"/></root>')
            ),
        )

        ocr.extract.assert_awaited_once()

    async def test_sparse_native_hierarchy_forces_ocr(self) -> None:
        """
        A NATIVE frame whose hierarchy parsed but yielded only a handful of
        container nodes is treated as incomplete; OCR must run to recover the
        text-bearing elements the dump missed.
        """

        ocr = self.__ocr_mock()
        await ScreenObservationService(ocr=ocr).observe(
            step_number=0,
            session_id="run-test",
            hashes=self.__hashes(),
            budget=self.__budget(),
            manifest=self.__manifest(),
            capture=self.__capture(
                xml_content='<root><android.widget.FrameLayout bounds="[0,0][2340,1080]"/></root>'
            ),
        )

        ocr.extract.assert_awaited_once()

    async def test_missing_xml_keeps_ocr_running(self) -> None:
        """
        Absent XML defaults to NATIVE; the existing OCR call path remains intact.
        """

        ocr = self.__ocr_mock()
        await ScreenObservationService(ocr=ocr).observe(
            step_number=0,
            session_id="run-test",
            hashes=self.__hashes(),
            budget=self.__budget(),
            manifest=self.__manifest(),
            capture=self.__capture(xml_content=None),
        )

        ocr.extract.assert_awaited_once()

    async def test_keyboard_visibility_default_when_no_device(self) -> None:
        """
        Sanity check the observation still returns a HIDDEN keyboard when no
        device adapter is wired. Belt-and-braces guard so the OCR-skip path
        doesn't inadvertently short-circuit unrelated branches.
        """

        ocr = self.__ocr_mock()

        observation = await ScreenObservationService(ocr=ocr).observe(
            step_number=0,
            session_id="run-test",
            hashes=self.__hashes(),
            budget=self.__budget(),
            manifest=self.__manifest(),
            capture=self.__capture(
                xml_content=('<root><android.webkit.WebView bounds="[0,0][2340,1080]"/></root>')
            ),
        )

        self.assertIs(observation.keyboard.visibility, KeyboardVisibility.UNKNOWN)
