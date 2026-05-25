from __future__ import annotations

import io
import unittest
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from PIL import Image

from fathom.constants.observation import KeyboardVisibility
from fathom.constants.state import CommonStateKey, CompletionReason, IntentStateKey
from fathom.schemas.actions import Bounds, CoordinateSource, CoordinateSystem
from fathom.schemas.observation import (
    ElementRole,
    ElementSource,
    KeyboardObservation,
    PerceivedElement,
    ScreenObservation,
)
from fathom.schemas.screens import ScreenCapture, ScreenHashBundle, ScreenState
from fathom.strategies.graph.intent.nodes.ground import GroundNode


class GroundNodeEarlyExitTest(unittest.IsolatedAsyncioTestCase):
    """
    Pins the GROUND node's three early-exit branches.

    GROUND is the entry point of the LangGraph cycle. Three conditions
    must terminate the run before the perception port is even consulted:
    cancellation, max-step cap, and an empty screenshot from the device.
    Each branch must call ``agent_state.mark_complete`` with the right
    :class:`CompletionReason` and return an ``IS_COMPLETE=True`` patch so
    the graph short-circuits without producing a downstream ANALYZE call.
    """

    @staticmethod
    def __provider(
        *,
        cancelled: bool = False,
        step_count: int = 0,
        max_steps: int = 20,
        image: bytes = b"PNG",
        width: int = 1000,
        height: int = 2000,
    ) -> MagicMock:
        """
        Mocked :class:`IntentNodeProvider` exposing only the surface area
        GROUND touches on the early-exit branches.

        ``is_cancelled`` is an :class:`AsyncMock` because the node awaits
        it; ``perceive`` is mocked so the empty-screenshot path can be
        forced by passing ``image=b""``.
        """

        provider = MagicMock(name="IntentNodeProvider")
        provider.is_cancelled = AsyncMock(return_value=cancelled)
        provider.context.workflow_id = "run-test"
        provider.context.agent_state.step_count = step_count
        provider.context.max_steps = max_steps
        provider.context.telemetry.info = AsyncMock()
        provider.context.telemetry.error = AsyncMock()
        provider.context.perception.perceive = AsyncMock(
            return_value=MagicMock(image=image, width=width, height=height, activity="app"),
        )
        provider.persistence.persist = MagicMock()
        return provider

    async def test_cancellation_terminates_with_cancelled_reason(self) -> None:
        """
        A cancelled run must mark complete with :attr:`CompletionReason.CANCELLED`
        and never reach the perception port.
        """

        provider = self.__provider(cancelled=True)
        node = GroundNode(provider=provider)

        result: Any = await node(state={})  # type: ignore[arg-type]

        self.assertTrue(result.get(CommonStateKey.IS_COMPLETE))
        self.assertEqual(
            result.get(CommonStateKey.COMPLETION_REASON),
            CompletionReason.CANCELLED.value,
        )
        provider.context.agent_state.mark_complete.assert_called_once_with(
            reason=CompletionReason.CANCELLED.value,
        )

    async def test_step_count_at_cap_terminates_with_max_steps_reason(self) -> None:
        """
        Reaching the configured step cap before planning the next action
        must terminate with :attr:`CompletionReason.MAX_STEPS`, not
        FAILED. The cap is checked before any work to avoid spending a
        capture on a step that cannot execute.
        """

        provider = self.__provider(cancelled=False, step_count=20, max_steps=20)
        node = GroundNode(provider=provider)

        result: Any = await node(state={})  # type: ignore[arg-type]

        self.assertTrue(result.get(CommonStateKey.IS_COMPLETE))
        self.assertEqual(
            result.get(CommonStateKey.COMPLETION_REASON),
            CompletionReason.MAX_STEPS.value,
        )
        provider.context.perception.perceive.assert_not_awaited()

    async def test_empty_capture_terminates_with_failed_reason(self) -> None:
        """
        An empty screenshot from the perception port is a hard failure
        (device disconnected, lost surface, etc.) and must terminate with
        :attr:`CompletionReason.FAILED` so the run is surfaced as broken
        rather than silently looping on empty captures.
        """

        provider = self.__provider(cancelled=False, image=b"")
        node = GroundNode(provider=provider)

        result: Any = await node(state={})  # type: ignore[arg-type]

        self.assertTrue(result.get(CommonStateKey.IS_COMPLETE))
        self.assertEqual(
            result.get(CommonStateKey.COMPLETION_REASON),
            CompletionReason.FAILED.value,
        )


class GroundNodeOcrManifestFallbackTest(unittest.IsolatedAsyncioTestCase):
    """
    Pins the XML-missing path where OCR must become the planner manifest.
    """

    @staticmethod
    def __png() -> bytes:
        """
        Build a valid PNG so perception overlays can render during the test.
        """

        image = Image.new("RGB", (1080, 2340), "white")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    @classmethod
    def __capture(cls) -> ScreenCapture:
        """
        Screenshot fixture with no XML hierarchy.
        """

        return ScreenCapture(
            width=1080,
            height=2340,
            activity="com.google.android.apps.nexuslauncher",
            image=cls.__png(),
            timestamp=123,
        )

    @staticmethod
    def __observation() -> ScreenObservation:
        """
        OCR-only observation matching the launcher Swiggy label.
        """

        return ScreenObservation(
            activity="com.google.android.apps.nexuslauncher",
            keyboard=KeyboardObservation(visibility=KeyboardVisibility.HIDDEN),
            hashes=ScreenHashBundle(
                visual_hash="c5ab006a70c02a85",
                xml_hash="",
                interaction_hash="",
            ),
            elements=(
                PerceivedElement(
                    identifier="ocr_13",
                    label_id="13",
                    text="Swiggy",
                    parent=None,
                    source=ElementSource.OCR,
                    role=ElementRole.TEXT,
                    confidence=0.96,
                    tappable=False,
                    bounds=Bounds(
                        x=284,
                        y=383,
                        width=108,
                        height=31,
                        source=CoordinateSource.OCR,
                        coordinate_system=CoordinateSystem.DEVICE_PIXEL,
                    ),
                ),
            ),
        )

    @classmethod
    def __provider(cls) -> MagicMock:
        """
        Provider mock exposing the full successful GROUND surface.
        """

        capture = cls.__capture()
        hashes = ScreenHashBundle(
            visual_hash="c5ab006a70c02a85",
            xml_hash="",
            interaction_hash="",
        )
        state = ScreenState(
            activity=capture.activity,
            timestamp=capture.timestamp,
            activity_hash="activityhash1234",
            visual_hash=hashes.visual_hash,
            xml_hash=hashes.xml_hash,
            interaction_hash=hashes.interaction_hash,
        )

        provider = MagicMock(name="IntentNodeProvider")
        provider.is_cancelled = AsyncMock(return_value=False)
        provider.context.workflow_id = "934671ae"
        provider.context.use_xml = True
        provider.context.max_steps = 50
        provider.context.agent_state.step_count = 0
        provider.context.agent_state.update_screen = MagicMock(return_value=True)
        provider.context.agent_state.runtime.screen.update = MagicMock()
        provider.context.telemetry.info = AsyncMock()
        provider.context.telemetry.error = AsyncMock()
        provider.context.metrics.record = MagicMock()
        provider.context.perception.perceive = AsyncMock(return_value=capture)
        provider.observer.resolve_capture_hashes.return_value = hashes
        provider.observer.build_screen_state.return_value = state
        provider.observer.observe = AsyncMock(return_value=cls.__observation())
        provider.persistence.persist = MagicMock()
        return provider

    async def test_ocr_elements_become_manifest_when_xml_is_absent(self) -> None:
        """
        Empty XML must not leave ``ELEMENTS`` empty when OCR found text.
        """

        provider = self.__provider()
        node = GroundNode(provider=provider)

        result: Any = await node(state={})  # type: ignore[arg-type]

        elements = result[IntentStateKey.ELEMENTS]
        self.assertEqual(elements["1"]["text"], "Swiggy")
        self.assertEqual(elements["1"]["source"], "ocr")
        self.assertEqual(elements["1"]["bounds"], "[284,383][392,414]")
        self.assertIsNotNone(result[CommonStateKey.CAPTURE].annotated_image)
