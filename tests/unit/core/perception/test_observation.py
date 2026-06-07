from __future__ import annotations

import unittest
from pathlib import Path
from typing import Tuple
from unittest.mock import AsyncMock, Mock

from fathom.constants.observation import KeyboardVisibility
from fathom.constants.ocr import OcrConfidence
from fathom.core.perception.observation import ScreenObservationService
from fathom.schemas.actions import Bounds, CoordinateSource, CoordinateSystem
from fathom.schemas.artifact import OcrPerceptionPayload, OcrRawPayload, OverlayPerceptionPayload
from fathom.schemas.budgets import PerceptionBudget
from fathom.schemas.observation import (
    ElementRole,
    ElementSource,
    OverlayObservation,
    PerceivedElement,
)
from fathom.schemas.ocr import OcrResult, OcrToken
from fathom.schemas.perception import KeyboardConfiguration, PerceptionConfiguration
from fathom.schemas.screens import ScreenCapture, ScreenHashBundle
from fathom.schemas.ui import LabeledElement, UIBounds


class _StaticOcr:
    """
    Test OCR port returning one stable token and raw provider JSON.
    """

    async def extract(self, *, capture: ScreenCapture, budget: PerceptionBudget) -> OcrResult:
        """
        Return a Swiggy OCR token regardless of capture contents.
        """

        _ = capture, budget
        return OcrResult(
            duration=12,
            raw_response='{"text": "Swiggy"}',
            tokens=(
                OcrToken(
                    text="Swiggy",
                    bounds=Bounds(
                        x=284,
                        y=383,
                        width=108,
                        height=31,
                        source=CoordinateSource.OCR,
                        coordinate_system=CoordinateSystem.DEVICE_PIXEL,
                    ),
                    raw_score=0.96,
                    confidence=OcrConfidence.HIGH,
                ),
            ),
        )


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

    async def test_oversized_manifest_bounds_are_clamped_before_schema_conversion(self) -> None:
        """
        Malformed provider bounds must be clipped to the visible viewport.
        """

        manifest = (
            LabeledElement(
                label="L1",
                bounds=UIBounds(x1=114, y1=888, x2=1092, y2=26643),
                attributes={
                    "type": "XCUIElementTypeTextView",
                    "value": "Privacy text",
                    "visible": "true",
                    "enabled": "true",
                    "scrollable": "true",
                    "axis": "vertical",
                    "kind": "viewport",
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

        self.assertEqual(len(observation.elements), 1)
        self.assertLessEqual(
            observation.elements[0].bounds.x + observation.elements[0].bounds.width,
            self.__capture().width,
        )
        self.assertLessEqual(
            observation.elements[0].bounds.y + observation.elements[0].bounds.height,
            self.__capture().height,
        )

    async def test_manifest_bounds_outside_viewport_are_dropped(self) -> None:
        """
        Elements fully outside the viewport must not enter observations.
        """

        manifest = (
            LabeledElement(
                label="L1",
                bounds=UIBounds(x1=1300, y1=3000, x2=1500, y2=3300),
                attributes={
                    "type": "XCUIElementTypeButton",
                    "value": "Invisible",
                    "visible": "true",
                    "enabled": "true",
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

        self.assertEqual(observation.elements, ())

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

        self.assertFalse(observation.keyboard.visibility is KeyboardVisibility.VISIBLE)

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

        self.assertFalse(observation.keyboard.visibility is KeyboardVisibility.VISIBLE)

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

    async def test_page_scroll_region_uses_logical_system_when_capture_dimensions_are_logical(
        self,
    ) -> None:
        """
        A synthetic page lane built from logical capture dimensions must
        stay logical so execution does not divide it by the retina scale.
        """

        observation = await ScreenObservationService().observe(
            capture=self.__capture().model_copy(update={"width": 402, "height": 874}),
            hashes=self.__hashes(),
            budget=self.__budget(),
            manifest=(),
            session_id="run-aekci",
            step_number=4,
        )

        self.assertEqual(len(observation.scroll), 1)
        self.assertEqual(observation.scroll[0].bounds.system, CoordinateSystem.LOGICAL)
        self.assertEqual(observation.scroll[0].bounds.width, 402)

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

    async def test_ocr_raw_and_annotated_artifacts_emit_when_ocr_contributes(self) -> None:
        """
        OCR enrichment must persist both raw provider JSON and OCR-only annotation.
        """

        pipeline = Mock()
        pipeline.emit = AsyncMock()

        observation = await ScreenObservationService(
            ocr=_StaticOcr(),
            pipeline=pipeline,
        ).observe(
            capture=self.__capture(),
            hashes=self.__hashes(),
            budget=self.__budget(),
            manifest=(),
            session_id="run-test",
            step_number=0,
        )

        self.assertEqual(observation.elements[0].text, "Swiggy")
        payloads = [call.kwargs["record"].payload for call in pipeline.emit.await_args_list]
        self.assertTrue(any(isinstance(payload, OcrRawPayload) for payload in payloads))
        self.assertTrue(any(isinstance(payload, OcrPerceptionPayload) for payload in payloads))
        raw_payloads = [payload for payload in payloads if isinstance(payload, OcrRawPayload)]
        self.assertEqual(raw_payloads[0].content, '{"text": "Swiggy"}')


class ScreenObservationServiceKeyboardProbeTest(unittest.IsolatedAsyncioTestCase):
    """
    Pin the device-backed fallback for keyboard detection in vision-only mode.
    """

    __FRAMES = Path(__file__).resolve().parents[3] / "fixtures" / "perception" / "frames"
    __IMAGE = __FRAMES / "home.png"

    @classmethod
    def __capture(cls) -> ScreenCapture:
        """
        Build one capture fixture from the on-disk Swiggy frame.
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
        Stable triple-hash fixture for the keyboard-probe tests.
        """

        return ScreenHashBundle(
            visual_hash="v" * 16,
            xml_hash="x" * 16,
            interaction_hash="i" * 16,
        )

    @staticmethod
    def __budget() -> PerceptionBudget:
        """
        Zero-budget so OCR/icon/overlay providers stay inert.
        """

        return PerceptionBudget(ocr=0, local=0, localization=0)

    async def test_device_probe_invoked_when_no_keyboard_element_present(self) -> None:
        """
        With no KEYBOARD-class manifest element, the service must consult the device adapter.
        """

        from fathom.schemas.observation import KeyboardObservation

        device = Mock()
        device.detect_keyboard = AsyncMock(
            return_value=KeyboardObservation(
                visibility=KeyboardVisibility.VISIBLE,
                bounds=Bounds(
                    x=0,
                    y=1507,
                    width=1080,
                    height=701,
                    coordinate_system=CoordinateSystem.DEVICE_PIXEL,
                ),
            )
        )

        observation = await ScreenObservationService(device=device).observe(
            capture=self.__capture(),
            hashes=self.__hashes(),
            budget=self.__budget(),
            manifest=(),
            session_id="run-test",
            step_number=0,
        )

        device.detect_keyboard.assert_awaited_once()
        self.assertIs(observation.keyboard.visibility, KeyboardVisibility.VISIBLE)
        self.assertIsNotNone(observation.keyboard.bounds)

    async def test_unknown_when_no_device_and_no_manifest_keyboard(self) -> None:
        """
        Without a device adapter, the service preserves the UNKNOWN sentinel.
        """

        observation = await ScreenObservationService().observe(
            capture=self.__capture(),
            hashes=self.__hashes(),
            budget=self.__budget(),
            manifest=(),
            session_id="run-test",
            step_number=0,
        )

        self.assertIs(observation.keyboard.visibility, KeyboardVisibility.UNKNOWN)

    async def test_device_probe_exception_degrades_to_unknown(self) -> None:
        """
        A throwing keyboard probe must not crash the observation pipeline; UNKNOWN is the fallback.
        """

        device = Mock()
        device.detect_keyboard = AsyncMock(side_effect=RuntimeError("transport closed"))

        observation = await ScreenObservationService(device=device).observe(
            capture=self.__capture(),
            hashes=self.__hashes(),
            budget=self.__budget(),
            manifest=(),
            session_id="run-test",
            step_number=0,
        )

        device.detect_keyboard.assert_awaited_once()
        self.assertIs(observation.keyboard.visibility, KeyboardVisibility.UNKNOWN)


class ScreenObservationServiceOcrTriggerTest(unittest.TestCase):
    """
    Pins the sparse-manifest gate that decides whether OCR enrichment runs.
    """

    @staticmethod
    def __element(*, text: str, identifier: str) -> PerceivedElement:
        """
        Build one perceived element carrying the supplied text label.
        """

        return PerceivedElement(
            parent=None,
            tappable=True,
            text=text,
            label_id=identifier,
            bounds=Bounds(
                x=0,
                y=0,
                width=100,
                height=40,
                source=CoordinateSource.XML,
                coordinate_system=CoordinateSystem.DEVICE_PIXEL,
            ),
            role=ElementRole.BUTTON,
            source=ElementSource.XML,
            identifier=identifier,
            confidence=1.0,
        )

    def test_unity_collapse_single_textful_element_triggers_ocr(self) -> None:
        """
        A manifest with one generic textful container fires OCR despite a 1.0 ratio.
        """

        elements = (self.__element(text="Game view", identifier="xml_1"),)

        decision = ScreenObservationService._ScreenObservationService__should_run_ocr(
            elements=elements,
        )

        self.assertTrue(decision)

    def test_two_textful_elements_still_below_count_floor(self) -> None:
        """
        Two text-bearing elements is still below the floor and must run OCR.
        """

        elements = (
            self.__element(text="Home", identifier="xml_1"),
            self.__element(text="Settings", identifier="xml_2"),
        )

        decision = ScreenObservationService._ScreenObservationService__should_run_ocr(
            elements=elements,
        )

        self.assertTrue(decision)

    def test_three_textful_elements_still_below_manifest_size_floor(self) -> None:
        """
        A three-element manifest is still too thin to trust the hierarchy; even
        with a perfect text ratio, OCR must run to cover the missing nodes.
        """

        elements = (
            self.__element(text="Home", identifier="xml_1"),
            self.__element(text="Settings", identifier="xml_2"),
            self.__element(text="Profile", identifier="xml_3"),
        )

        decision = ScreenObservationService._ScreenObservationService__should_run_ocr(
            elements=elements,
        )

        self.assertTrue(decision)

    def test_dense_textful_manifest_skips_ocr(self) -> None:
        """
        Once the manifest is dense and well-labelled (size, text-bearing count,
        and coverage all above their floors), OCR is redundant and may skip.
        """

        elements = tuple(
            self.__element(text=f"Item {index}", identifier=f"xml_{index}") for index in range(8)
        )

        decision = ScreenObservationService._ScreenObservationService__should_run_ocr(
            elements=elements,
        )

        self.assertFalse(decision)

    def test_low_ratio_manifest_still_triggers_ocr(self) -> None:
        """
        Even with enough text-bearing elements, a low ratio still triggers OCR.
        """

        textful = tuple(
            self.__element(text=f"Item {index}", identifier=f"xml_{index}") for index in range(3)
        )
        empty = tuple(
            self.__element(text="", identifier=f"xml_empty_{index}") for index in range(20)
        )
        elements = textful + empty

        decision = ScreenObservationService._ScreenObservationService__should_run_ocr(
            elements=elements,
        )

        self.assertTrue(decision)
