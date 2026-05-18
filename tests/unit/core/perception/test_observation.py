from __future__ import annotations

import unittest
from pathlib import Path
from typing import Tuple

from fathom.core.perception.observation import ScreenObservationService
from fathom.schemas.budgets import PerceptionBudget
from fathom.schemas.observation import ElementSource, OverlayObservation
from fathom.schemas.screens import ScreenCapture, ScreenHashBundle
from fathom.schemas.ui import LabeledElement, UIBounds


class ScreenObservationServiceOverlayDedupTest(unittest.IsolatedAsyncioTestCase):
    """
    Pins the overlay-dedup invariant on :class:`ScreenObservationService`.

    The supervisor only needs a single BLOCKING vote per screen. Two
    qualifying overlay elements in the same manifest (stacked dialogs)
    must therefore collapse into one :class:`OverlayObservation`, not be
    surfaced as two independent signals. The fixture PNG is a real
    1206x2622 Swiggy capture from ``assets/screenshot/`` — only the byte
    payload is consumed.
    """

    __FRAMES = Path(__file__).resolve().parents[3] / "fixtures" / "perception" / "frames"
    __IMAGE = __FRAMES / "home.png"

    @classmethod
    def __capture(cls) -> ScreenCapture:
        """
        :class:`ScreenCapture` fixture wrapping the on-disk PNG. The width
        and height match the source asset so any subsequent visual
        processing reads consistent pixel coordinates.
        """

        return ScreenCapture(
            width=1206,
            height=2622,
            activity="app",
            image=cls.__IMAGE.read_bytes(),
            timestamp=0,
        )

    @staticmethod
    def __hashes() -> ScreenHashBundle:
        """
        Deterministic :class:`ScreenHashBundle` fixture. The observation
        service does not read these values; they exist only to satisfy
        the schema.
        """

        return ScreenHashBundle(
            visual_hash="0" * 16,
            xml_hash="a" * 16,
            interaction_hash="b" * 16,
        )

    @staticmethod
    def __budget() -> PerceptionBudget:
        """
        Zero-budget :class:`PerceptionBudget`. The test exercises only the
        manifest-driven overlay path, so OCR / icon / overlay providers
        must stay inert; a zero budget keeps the noop fall-back active.
        """

        return PerceptionBudget(ocr=0, local=0, localization=0)

    @staticmethod
    def __overlay_manifest() -> Tuple[LabeledElement, ...]:
        """
        Two large model-sourced dialog elements. Both individually meet
        the overlay-area threshold; the test pins that the service still
        produces exactly one :class:`OverlayObservation`.
        """

        return (
            LabeledElement(
                label="L1",
                bounds=UIBounds(x1=0, y1=0, x2=1206, y2=2400),
                attributes={
                    "class": "DialogContainer",
                    "source": "model",
                    "text": "First overlay",
                },
            ),
            LabeledElement(
                label="L2",
                bounds=UIBounds(x1=50, y1=50, x2=1156, y2=2350),
                attributes={
                    "class": "DialogContainer",
                    "source": "model",
                    "text": "Second overlay",
                },
            ),
        )

    async def test_stacked_qualifying_elements_collapse_to_single_overlay(self) -> None:
        """
        Two stacked qualifying elements must yield exactly one overlay
        observation. The supervisor consumes overlays as boolean BLOCKING
        votes, so reporting both would double-count the same screen state.
        """

        observation = await ScreenObservationService().observe(
            capture=self.__capture(),
            hashes=self.__hashes(),
            budget=self.__budget(),
            manifest=self.__overlay_manifest(),
            session_id="run-test",
            step_number=0,
        )

        self.assertEqual(len(observation.overlays), 1)
        self.assertIsInstance(observation.overlays[0], OverlayObservation)

    async def test_cv_attribute_maps_to_cv_element_source(self) -> None:
        """
        A :class:`LabeledElement` carrying ``attributes["source"] == "cv"``
        — produced by :class:`VisualControlLabeler` — must surface in the
        resulting :class:`ScreenObservation` with ``ElementSource.CV``,
        not ``ElementSource.VISION``. Mis-routing this tag back to
        VISION poisons downstream telemetry (a real iOS run was mis-
        reporting CV-only buttons as vision-localized).
        """

        manifest = (
            LabeledElement(
                label="L1",
                bounds=UIBounds(x1=200, y1=400, x2=900, y2=520),
                attributes={
                    "class": "VisualControl",
                    "source": "cv",
                    "confidence": "0.85",
                },
            ),
        )

        observation = await ScreenObservationService().observe(
            capture=self.__capture(),
            hashes=self.__hashes(),
            budget=self.__budget(),
            manifest=manifest,
            session_id="run-test",
            step_number=0,
        )

        cv_elements = tuple(
            element for element in observation.elements if element.source == ElementSource.CV
        )
        self.assertGreaterEqual(len(cv_elements), 1)
        self.assertFalse(
            any(
                element.source == ElementSource.VISION
                for element in observation.elements
                if "VisualControl" in str(element.identifier) or element.label_id == "L1"
            ),
            "CV-tagged labeled elements must not be rewritten to ElementSource.VISION.",
        )

    async def test_small_xml_buttons_do_not_qualify_as_overlay(self) -> None:
        """
        XML-sourced elements only qualify as overlays when their role is
        ``OVERLAY`` explicitly. A small XML button — even one named
        ``Close`` — must not surface as a blocking layer.
        """

        manifest = (
            LabeledElement(
                label="L1",
                bounds=UIBounds(x1=20, y1=20, x2=200, y2=80),
                attributes={"class": "Button", "source": "xml", "text": "Close"},
            ),
        )

        observation = await ScreenObservationService().observe(
            capture=self.__capture(),
            hashes=self.__hashes(),
            budget=self.__budget(),
            manifest=manifest,
            session_id="run-test",
            step_number=0,
        )

        self.assertEqual(observation.overlays, ())
