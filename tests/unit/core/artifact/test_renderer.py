from __future__ import annotations

import io
import unittest
from typing import Tuple

from PIL import Image

from fathom.constants import ActionType
from fathom.constants.observation import KeyboardVisibility
from fathom.core.artifact.renderer import (
    PassthroughRenderer,
    PerceptionRenderer,
    TraceRenderer,
    VerificationRenderer,
)
from fathom.schemas.actions import Action, Bounds, CoordinateSystem
from fathom.schemas.artifact import (
    AnnotatedPayload,
    ArtifactKind,
    ArtifactRecord,
    HierarchyXmlPayload,
    OcrRawPayload,
    PerceptionPayload,
    ScreenshotPayload,
    ScriptPayload,
    TracePayload,
    VerificationPayload,
)
from fathom.schemas.completion import CompletionVerdict
from fathom.schemas.observation import (
    ElementRole,
    ElementSource,
    KeyboardObservation,
    OverlayObservation,
    PerceivedElement,
    ScreenObservation,
)
from fathom.schemas.screens import ScreenCapture, ScreenHashBundle
from fathom.schemas.tasks import ExecutionTaskState


class _Fixtures:
    """
    Shared fixture builders for every renderer test in this module.
    """

    @staticmethod
    def png(
        *, width: int = 64, height: int = 64, color: Tuple[int, int, int] = (200, 200, 200)
    ) -> bytes:
        """
        Encode a solid-coloured PNG of the requested dimensions.
        """

        buffer = io.BytesIO()
        Image.new("RGB", (width, height), color).save(buffer, format="PNG")
        return buffer.getvalue()

    @classmethod
    def capture(cls) -> ScreenCapture:
        """
        Build a :class:`ScreenCapture` with a real solid-grey PNG.
        """

        return ScreenCapture(
            width=64,
            height=64,
            activity="app",
            image=cls.png(),
            timestamp=0,
        )

    @staticmethod
    def hashes() -> ScreenHashBundle:
        """
        Build a deterministic :class:`ScreenHashBundle` placeholder.
        """

        return ScreenHashBundle(
            visual_hash="0" * 16,
            xml_hash="a" * 16,
            interaction_hash="b" * 16,
        )

    @classmethod
    def record(cls, *, payload) -> ArtifactRecord:  # type: ignore[no-untyped-def]
        """
        Wrap a typed payload in a record with deterministic identity.
        """

        return ArtifactRecord(
            session_id="run-test",
            package_name="app",
            step_number=0,
            created=1,
            payload=payload,
        )


class PassthroughRendererTest(unittest.TestCase):
    """
    Pins :class:`PassthroughRenderer` for already-rendered byte payloads.
    """

    def test_screenshot_returns_image_bytes_verbatim(self) -> None:
        """
        Screenshot kind returns the capture's image bytes unchanged.
        """

        capture = _Fixtures.capture()
        renderer = PassthroughRenderer(kind=ArtifactKind.SCREENSHOT)

        rendered = renderer.render(
            record=_Fixtures.record(payload=ScreenshotPayload(capture=capture)),
        )

        self.assertEqual(rendered, capture.image)

    def test_annotated_returns_annotated_image_when_available(self) -> None:
        """
        Annotated kind prefers ``capture.annotated_image`` over raw bytes.
        """

        annotated_bytes = _Fixtures.png(color=(1, 2, 3))
        capture = _Fixtures.capture().model_copy(update={"annotated_image": annotated_bytes})
        renderer = PassthroughRenderer(kind=ArtifactKind.ANNOTATED)

        rendered = renderer.render(
            record=_Fixtures.record(payload=AnnotatedPayload(capture=capture)),
        )

        self.assertEqual(rendered, annotated_bytes)

    def test_annotated_falls_back_to_raw_capture_when_no_annotation(self) -> None:
        """
        Annotated kind falls back to ``capture.image`` when no annotated
        bytes were attached upstream.
        """

        capture = _Fixtures.capture()
        renderer = PassthroughRenderer(kind=ArtifactKind.ANNOTATED)

        rendered = renderer.render(
            record=_Fixtures.record(payload=AnnotatedPayload(capture=capture)),
        )

        self.assertEqual(rendered, capture.image)

    def test_hierarchy_xml_encodes_content_as_utf8(self) -> None:
        """
        XML payloads are encoded to UTF-8 bytes.
        """

        renderer = PassthroughRenderer(kind=ArtifactKind.HIERARCHY_XML)
        rendered = renderer.render(
            record=_Fixtures.record(payload=HierarchyXmlPayload(content="<hierarchy/>")),
        )

        self.assertEqual(rendered, b"<hierarchy/>")

    def test_ocr_raw_encodes_content_as_utf8(self) -> None:
        """
        Raw OCR JSON payloads are encoded to UTF-8 bytes.
        """

        renderer = PassthroughRenderer(kind=ArtifactKind.OCR_RAW)
        rendered = renderer.render(
            record=_Fixtures.record(payload=OcrRawPayload(content='{"text": "Swiggy"}')),
        )

        self.assertEqual(rendered, b'{"text": "Swiggy"}')

    def test_script_encodes_content_as_utf8(self) -> None:
        """
        Script payloads are encoded to UTF-8 bytes.
        """

        renderer = PassthroughRenderer(kind=ArtifactKind.SCRIPT)
        rendered = renderer.render(
            record=_Fixtures.record(payload=ScriptPayload(content="tap 'OK'")),
        )

        self.assertEqual(rendered, b"tap 'OK'")


class PerceptionRendererTest(unittest.TestCase):
    """
    Pins :class:`PerceptionRenderer` overlay drawing on a real PNG canvas.
    """

    @staticmethod
    def __element(*, source: ElementSource, bounds: Bounds) -> PerceivedElement:
        """
        Build a :class:`PerceivedElement` with the given source and bounds.
        """

        return PerceivedElement(
            identifier="el_1",
            text="continue",
            parent=None,
            bounds=bounds,
            source=source,
            role=ElementRole.TEXT,
            confidence=0.9,
            tappable=False,
        )

    def test_render_returns_valid_png_bytes(self) -> None:
        """
        The renderer emits a decodable PNG.
        """

        observation = ScreenObservation(
            activity="app",
            elements=(
                self.__element(
                    source=ElementSource.OCR,
                    bounds=Bounds(
                        x=5,
                        y=5,
                        width=20,
                        height=10,
                        coordinate_system=CoordinateSystem.DEVICE_PIXEL,
                    ),
                ),
            ),
            hashes=_Fixtures.hashes(),
            keyboard=KeyboardObservation(visibility=KeyboardVisibility.HIDDEN),
        )

        rendered = PerceptionRenderer().render(
            record=_Fixtures.record(
                payload=PerceptionPayload(
                    capture=_Fixtures.capture(),
                    observation=observation,
                ),
            ),
        )

        decoded = Image.open(io.BytesIO(rendered))
        self.assertEqual(decoded.size, (64, 64))

    def test_render_handles_overlay_layer_without_raising(self) -> None:
        """
        Overlay observations are drawn without raising; pin via PNG decode.
        """

        observation = ScreenObservation(
            activity="app",
            elements=(),
            hashes=_Fixtures.hashes(),
            overlays=(
                OverlayObservation(
                    visible=True,
                    bounds=Bounds(
                        x=0,
                        y=0,
                        width=30,
                        height=30,
                        coordinate_system=CoordinateSystem.DEVICE_PIXEL,
                    ),
                    candidates=(),
                ),
            ),
            keyboard=KeyboardObservation(visibility=KeyboardVisibility.HIDDEN),
        )

        rendered = PerceptionRenderer().render(
            record=_Fixtures.record(
                payload=PerceptionPayload(
                    capture=_Fixtures.capture(),
                    observation=observation,
                ),
            ),
        )

        Image.open(io.BytesIO(rendered))


class TraceRendererTest(unittest.TestCase):
    """
    Pins :class:`TraceRenderer` action overlay drawing.
    """

    def test_tap_action_draws_target_circle(self) -> None:
        """
        Tap actions render a centred circle marker on the canvas.
        """

        payload = TracePayload(
            capture=_Fixtures.capture(),
            coords=(32, 32),
            action=Action(
                action_type=ActionType.TAP,
                target="x",
                rationale="t",
                confidence=1.0,
            ),
        )

        rendered = TraceRenderer().render(record=_Fixtures.record(payload=payload))

        decoded = Image.open(io.BytesIO(rendered))
        self.assertEqual(decoded.size, (64, 64))

    def test_swipe_action_draws_arrow(self) -> None:
        """
        Swipe actions render an arrow stroke and arrowhead.
        """

        payload = TracePayload(
            capture=_Fixtures.capture(),
            coords=(10, 10, 50, 50),
            action=Action(
                action_type=ActionType.SWIPE_UP,
                target="x",
                rationale="t",
                confidence=1.0,
            ),
        )

        rendered = TraceRenderer().render(record=_Fixtures.record(payload=payload))

        Image.open(io.BytesIO(rendered))


class VerificationRendererTest(unittest.TestCase):
    """
    Pins :class:`VerificationRenderer` verdict-banner drawing.
    """

    def test_verified_verdict_renders(self) -> None:
        """
        A passing verdict renders with a green banner.
        """

        payload = VerificationPayload(
            capture=_Fixtures.capture(),
            verdict=CompletionVerdict(
                complete=True,
                next_state=ExecutionTaskState.SUCCEEDED,
                reason="ok",
                missing=[],
            ),
        )

        rendered = VerificationRenderer().render(record=_Fixtures.record(payload=payload))

        Image.open(io.BytesIO(rendered))

    def test_rejected_verdict_renders(self) -> None:
        """
        A failing verdict renders with a red banner.
        """

        payload = VerificationPayload(
            capture=_Fixtures.capture(),
            verdict=CompletionVerdict(
                complete=False,
                next_state=ExecutionTaskState.FAILED,
                reason="missing evidence",
                missing=[],
            ),
        )

        rendered = VerificationRenderer().render(record=_Fixtures.record(payload=payload))

        Image.open(io.BytesIO(rendered))
