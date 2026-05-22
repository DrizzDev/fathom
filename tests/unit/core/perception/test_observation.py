from __future__ import annotations

import unittest
from pathlib import Path
from typing import Tuple
from unittest.mock import AsyncMock, Mock

from fathom.core.perception.observation import ScreenObservationService
from fathom.schemas.artifact import OverlayPerceptionPayload
from fathom.schemas.budgets import PerceptionBudget
from fathom.schemas.observation import ElementSource, OverlayObservation
from fathom.schemas.perception import KeyboardConfiguration, PerceptionConfiguration
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

    async def test_visual_controls_are_not_injected_when_cv_is_disabled(self) -> None:
        """
        CV-derived visual controls must stay out of observations when the
        runtime CV toggle is disabled.
        """

        observation = await ScreenObservationService(
            configuration=PerceptionConfiguration(),
        ).observe(
            capture=self.__capture(),
            hashes=self.__hashes(),
            budget=self.__budget(),
            manifest=(),
            session_id="run-test",
            step_number=0,
        )

        self.assertEqual(
            tuple(
                element for element in observation.elements if element.source == ElementSource.CV
            ),
            (),
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

    async def test_bottom_controls_do_not_imply_keyboard_visibility(self) -> None:
        """
        Large bottom controls are not enough to claim a visible keyboard.
        """

        manifest = (
            LabeledElement(
                label="L1",
                bounds=UIBounds(x1=0, y1=2200, x2=1206, y2=2480),
                attributes={
                    "class": "BottomActionBar",
                    "source": "model",
                    "text": "Apply coupon",
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

        self.assertFalse(observation.keyboard.visible)

    async def test_ios_hierarchy_disables_visual_keyboard_fallback(self) -> None:
        """
        iOS hierarchy-backed captures must not hallucinate a keyboard from bottom-screen texture.
        """

        capture = self.__capture().model_copy(
            update={
                "xml_content": (
                    "<XCUIElementTypeApplication>"
                    "<XCUIElementTypeSearchField visible='true'/>"
                    "</XCUIElementTypeApplication>"
                )
            }
        )

        observation = await ScreenObservationService().observe(
            capture=capture,
            hashes=self.__hashes(),
            budget=self.__budget(),
            manifest=(),
            session_id="run-test",
            step_number=0,
        )

        self.assertFalse(observation.keyboard.visible)

    async def test_keyboard_detection_stays_off_when_disabled(self) -> None:
        """
        Explicit keyboard-like manifest elements must be ignored when keyboard detection is disabled.
        """

        manifest = (
            LabeledElement(
                label="K1",
                bounds=UIBounds(x1=0, y1=1800, x2=1206, y2=2622),
                attributes={"class": "KeyboardView", "source": "xml"},
            ),
        )

        observation = await ScreenObservationService(
            configuration=PerceptionConfiguration(
                keyboard=KeyboardConfiguration(enabled=False),
            )
        ).observe(
            capture=self.__capture(),
            hashes=self.__hashes(),
            budget=self.__budget(),
            manifest=manifest,
            session_id="run-test",
            step_number=0,
        )

        self.assertFalse(observation.keyboard.visible)

    async def test_page_scroll_region_is_inferred_when_only_nested_strip_scrollview_exists(
        self,
    ) -> None:
        """
        A feed-like screen with only small nested scrollviews should still expose a page scroll lane.
        """

        manifest = (
            LabeledElement(
                label="search",
                bounds=UIBounds(x1=16, y1=221, x2=386, y2=278),
                attributes={
                    "type": "XCUIElementTypeSearchField",
                    "source": "xml",
                    "label": "Search, Order, Enjoy, Repeat!",
                },
            ),
            LabeledElement(
                label="chip_row",
                bounds=UIBounds(x1=16, y1=741, x2=386, y2=782),
                attributes={
                    "type": "XCUIElementTypeScrollView",
                    "source": "xml",
                },
            ),
            LabeledElement(
                label="nav_1",
                bounds=UIBounds(x1=40, y1=2487, x2=220, y2=2622),
                attributes={"type": "XCUIElementTypeButton", "source": "xml", "label": "Food"},
            ),
            LabeledElement(
                label="nav_2",
                bounds=UIBounds(x1=240, y1=2487, x2=420, y2=2622),
                attributes={"type": "XCUIElementTypeButton", "source": "xml", "label": "Bolt"},
            ),
            LabeledElement(
                label="nav_3",
                bounds=UIBounds(x1=440, y1=2487, x2=620, y2=2622),
                attributes={"type": "XCUIElementTypeButton", "source": "xml", "label": "99 store"},
            ),
            LabeledElement(
                label="nav_4",
                bounds=UIBounds(x1=640, y1=2487, x2=820, y2=2622),
                attributes={"type": "XCUIElementTypeButton", "source": "xml", "label": "EatRight"},
            ),
            LabeledElement(
                label="nav_5",
                bounds=UIBounds(x1=840, y1=2487, x2=1020, y2=2622),
                attributes={"type": "XCUIElementTypeButton", "source": "xml", "label": "Reorder"},
            ),
        )

        observation = await ScreenObservationService().observe(
            capture=self.__capture().model_copy(
                update={"xml_content": "<XCUIElementTypeApplication></XCUIElementTypeApplication>"}
            ),
            hashes=self.__hashes(),
            budget=self.__budget(),
            manifest=manifest,
            session_id="run-test",
            step_number=0,
        )

        page_regions = tuple(
            region for region in observation.scroll if region.bounds.height >= int(2622 * 0.30)
        )
        self.assertGreaterEqual(len(page_regions), 1)

    async def test_large_manifest_container_becomes_explicit_scroll_region(self) -> None:
        """
        A large manifest-backed container should surface as an explicit scroll region.
        """

        manifest = (
            LabeledElement(
                label="feed",
                bounds=UIBounds(x1=0, y1=330, x2=1206, y2=2200),
                attributes={"class": "Cell", "source": "xml"},
            ),
            LabeledElement(
                label="card_1",
                bounds=UIBounds(x1=48, y1=420, x2=1158, y2=1300),
                attributes={"class": "Other", "source": "xml", "label": "Restaurant A"},
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

        explicit = tuple(
            region
            for region in observation.scroll
            if region.manifest_label_id is not None and region.bounds.width == 1206
        )
        self.assertTrue(explicit)
        self.assertEqual(explicit[0].manifest_label_id, "feed")

    async def test_horizontal_scroll_regions_do_not_also_infer_vertical_page_region(self) -> None:
        """
        Explicit horizontal scroll containers must not be mixed with a synthetic vertical page region.
        """

        manifest = (
            LabeledElement(
                label="carousel",
                bounds=UIBounds(x1=0, y1=420, x2=1206, y2=620),
                attributes={
                    "type": "XCUIElementTypeScrollView",
                    "source": "xml",
                    "axis": "horizontal",
                    "kind": "carousel",
                    "scrollable": "true",
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

        self.assertEqual(len(observation.scroll), 1)
        self.assertEqual(observation.scroll[0].axis, "horizontal")
        self.assertEqual(observation.scroll[0].manifest_label_id, "carousel")
        self.assertIsNone(observation.scroll[0].observation_region_id)

    async def test_manifest_only_observation_does_not_emit_merged_perception_artifact(self) -> None:
        """
        Manifest/accessibility-only observation should not create a duplicate merged perception image.
        """

        pipeline = Mock()
        pipeline.emit = AsyncMock()
        manifest = (
            LabeledElement(
                label="L1",
                bounds=UIBounds(x1=20, y1=20, x2=200, y2=80),
                attributes={"class": "Button", "source": "xml", "text": "Close"},
            ),
        )

        await ScreenObservationService(pipeline=pipeline).observe(
            capture=self.__capture(),
            hashes=self.__hashes(),
            budget=self.__budget(),
            manifest=manifest,
            session_id="run-test",
            step_number=0,
        )

        self.assertEqual(
            [type(call.kwargs["record"].payload) for call in pipeline.emit.await_args_list],
            [],
        )

    async def test_overlay_artifact_still_emits_without_merged_perception_image(self) -> None:
        """
        Overlay-specific artifacts should still be emitted when overlay evidence exists.
        """

        pipeline = Mock()
        pipeline.emit = AsyncMock()

        await ScreenObservationService(pipeline=pipeline).observe(
            capture=self.__capture(),
            hashes=self.__hashes(),
            budget=self.__budget(),
            manifest=self.__overlay_manifest(),
            session_id="run-test",
            step_number=0,
        )

        payload_types = [
            type(call.kwargs["record"].payload) for call in pipeline.emit.await_args_list
        ]
        self.assertEqual(payload_types, [OverlayPerceptionPayload])
