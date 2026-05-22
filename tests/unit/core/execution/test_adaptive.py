from __future__ import annotations

import unittest
from pathlib import Path
from typing import Optional, Tuple
from unittest.mock import patch

from fathom.constants.interaction import SwipeSpeed
from fathom.constants.scroll import (
    ScrollDirection,
    ScrollEvidenceSource,
    ScrollVerdictKind,
    SurfaceKind,
)
from fathom.core.execution.scroll import AdaptiveScrollSupervisor
from fathom.core.execution.scroll import ScrollPlanner as AdaptiveScrollPlanner
from fathom.interfaces.device import DevicePort
from fathom.interfaces.scroll import ScrollDetectPort, ScrollSurfacePort
from fathom.schemas.actions import (
    Bounds,
    CoordinateSource,
    CoordinateSystem,
    ExecutionRegion,
    GesturePath,
)
from fathom.schemas.command import CommandScopeKind
from fathom.schemas.configuration import DeviceRuntimeConfiguration
from fathom.schemas.observation import (
    ElementRole,
    ElementSource,
    KeyboardObservation,
    PerceivedElement,
    ScreenObservation,
)
from fathom.schemas.results import ActionResult
from fathom.schemas.screens import ScreenCapture, ScreenHashBundle
from fathom.schemas.scroll import ScrollContext, ScrollScope, ScrollSurface, ScrollVerdict
from fathom.utils.coordinates import CoordinateConverter

PROJECT_ROOT = Path(__file__).resolve().parents[4]


class AdaptiveScrollSupervisorTest(unittest.IsolatedAsyncioTestCase):
    """
    Covers bounded adaptation away from the real promo-band geometry.
    """

    async def test_no_progress_uses_bounded_second_attempt_when_footer_blocks_current_start(
        self,
    ) -> None:
        """
        Retry once in-execute and shift away from the original lane and sticky footer band.
        """

        before = self.__capture(name="before.png")
        device = FakeDevice(after=before.image)
        detector = FakeDetect(
            verdicts=(
                ScrollVerdict(
                    kind=ScrollVerdictKind.NO_PROGRESS,
                    source=ScrollEvidenceSource.CORRELATION,
                    confidence=0.99,
                    distance=0,
                    detail="stuck_on_promo",
                ),
                ScrollVerdict(
                    kind=ScrollVerdictKind.NO_PROGRESS,
                    source=ScrollEvidenceSource.CORRELATION,
                    confidence=0.99,
                    distance=0,
                    detail="stuck_on_promo_retry",
                ),
            )
        )
        surface = FakeSurface(
            hints=(
                ScrollSurface(
                    kind=SurfaceKind.FOOTER,
                    bounds=Bounds(
                        x=0,
                        y=1881,
                        width=1080,
                        height=459,
                        coordinate_system=CoordinateSystem.DEVICE_PIXEL,
                    ),
                    source=ScrollEvidenceSource.SURFACE,
                    detail="bottom_navigation",
                ),
            )
        )

        supervisor = AdaptiveScrollSupervisor(
            device=device,
            detector=detector,
            surface=surface,
        )
        converter = CoordinateConverter(logical_width=1080, logical_height=2340)
        region = converter.viewport_region()
        current = converter.resolve_scroll_path(region=region, direction="down")

        result, outcome, trace_events = await supervisor.execute(
            before=before,
            observation=self.__observation(),
            context=ScrollContext(direction=ScrollDirection.DOWN, region=region),
            current=current,
            converter=converter,
            policy=DeviceRuntimeConfiguration().interaction.policy.scroll.adaptive.model_copy(
                update={"enabled": True, "maximum_attempts": 3}
            ),
        )

        self.assertFalse(result.success)
        self.assertFalse(outcome.success)
        self.assertEqual(len(device.swipes), 2)
        self.assertLess(device.swipes[0][1], 1881)
        self.assertLess(device.swipes[1][1], 1881)
        self.assertEqual(device.swipes[0][5], SwipeSpeed.SLOW)
        self.assertNotEqual(device.swipes[0][:4], device.swipes[1][:4])

    async def test_evaluates_single_attempt_against_its_pre_attempt_screen(self) -> None:
        """
        Compare the dispatched attempt against the screenshot that existed just before it.
        """

        before = ScreenCapture(
            width=1080,
            height=2340,
            activity="com.test.app",
            image=b"before",
            timestamp=1,
        )
        device = FakeDevice(after_sequence=(b"after_one", b"after_two"))
        detector = RecordingDetect(
            verdicts=(
                ScrollVerdict(
                    kind=ScrollVerdictKind.NO_PROGRESS,
                    source=ScrollEvidenceSource.CORRELATION,
                    confidence=0.99,
                    distance=0,
                    detail="no_move",
                ),
                ScrollVerdict(
                    kind=ScrollVerdictKind.NO_PROGRESS,
                    source=ScrollEvidenceSource.CORRELATION,
                    confidence=0.99,
                    distance=0,
                    detail="no_move_retry",
                ),
            )
        )

        supervisor = AdaptiveScrollSupervisor(
            device=device,
            detector=detector,
            surface=FakeSurface(hints=()),
        )
        converter = CoordinateConverter(logical_width=1080, logical_height=2340)
        region = converter.viewport_region()
        current = converter.resolve_scroll_path(region=region, direction="down")

        result, outcome, trace_events = await supervisor.execute(
            before=before,
            observation=self.__observation(),
            context=ScrollContext(direction=ScrollDirection.DOWN, region=region),
            current=current,
            converter=converter,
            policy=DeviceRuntimeConfiguration().interaction.policy.scroll.adaptive.model_copy(
                update={"enabled": True, "maximum_attempts": 3}
            ),
        )

        self.assertFalse(result.success)
        self.assertEqual(len(outcome.attempts), 2)
        self.assertEqual(detector.before_images, [b"before", b"after_one"])
        self.assertEqual(trace_events[0].capture.image, b"before")
        self.assertEqual(trace_events[1].capture.image, b"after_one")

    async def test_refreshes_post_attempt_capture_metadata_from_device_snapshot(self) -> None:
        """
        Post-attempt observation should use the fresh snapshot state, not stale pre-attempt metadata.
        """

        before = ScreenCapture(
            width=1080,
            height=2340,
            activity="com.test.before",
            image=b"before",
            xml_content="<before />",
            timestamp=1,
            metadata={"capture_duration": 0.1},
        )
        device = FakeDevice(after_sequence=(b"after_one",))
        detector = RecordingDetect(
            verdicts=(
                ScrollVerdict(
                    kind=ScrollVerdictKind.NO_PROGRESS,
                    source=ScrollEvidenceSource.CORRELATION,
                    confidence=0.99,
                    distance=0,
                    detail="no_move",
                ),
            )
        )
        supervisor = AdaptiveScrollSupervisor(
            device=device,
            detector=detector,
            surface=FakeSurface(hints=()),
        )
        converter = CoordinateConverter(logical_width=1080, logical_height=2340)
        region = converter.viewport_region()
        current = converter.resolve_scroll_path(region=region, direction="down")

        await supervisor.execute(
            before=before,
            observation=self.__observation(),
            context=ScrollContext(direction=ScrollDirection.DOWN, region=region),
            current=current,
            converter=converter,
            policy=DeviceRuntimeConfiguration().interaction.policy.scroll.adaptive.model_copy(
                update={"enabled": True, "maximum_attempts": 1}
            ),
        )

        self.assertEqual(detector.after_activities, ["com.aranoah.healthkart.plus"])
        self.assertEqual(detector.after_xml, [None])

    async def test_moves_start_point_out_of_occupied_content_when_gap_exists(self) -> None:
        """
        Choose a clear drag point instead of starting inside a content card.
        """

        before = self.__capture(name="before.png")
        device = FakeDevice(after=before.image)
        detector = FakeDetect(
            verdicts=(
                ScrollVerdict(
                    kind=ScrollVerdictKind.NO_PROGRESS,
                    source=ScrollEvidenceSource.CORRELATION,
                    confidence=0.99,
                    distance=0,
                    detail="stuck",
                ),
            )
        )
        supervisor = AdaptiveScrollSupervisor(
            device=device,
            detector=detector,
            surface=FakeSurface(hints=()),
        )
        converter = CoordinateConverter(logical_width=1080, logical_height=2340)
        region = converter.region_from_bounds(
            bounds=Bounds(
                x=0,
                y=393,
                width=1080,
                height=1390,
                coordinate_system=CoordinateSystem.DEVICE_PIXEL,
            ),
            source=CoordinateSource.MODEL,
        )
        current = converter.resolve_scroll_path(region=region, direction="down")

        await supervisor.execute(
            before=before,
            observation=self.__occupied_observation(),
            context=ScrollContext(direction=ScrollDirection.DOWN, region=region),
            current=current,
            converter=converter,
            policy=DeviceRuntimeConfiguration().interaction.policy.scroll.adaptive.model_copy(
                update={
                    "enabled": True,
                    "maximum_attempts": 1,
                    "suspicious_bottom_ratio": 1.0,
                }
            ),
        )

        self.assertEqual(len(device.swipes), 1)
        start_y = device.swipes[0][1]
        self.assertTrue(start_y < 1650 or start_y > 1830)
        self.assertGreaterEqual(abs(device.swipes[0][1] - device.swipes[0][3]), 260)

    async def test_preserves_travel_when_corridor_is_clear(self) -> None:
        """
        Keep a long swipe long when the corridor does not force trimming.
        """

        before = self.__capture(name="before.png")
        device = FakeDevice(after=before.image)
        detector = FakeDetect(
            verdicts=(
                ScrollVerdict(
                    kind=ScrollVerdictKind.NO_PROGRESS,
                    source=ScrollEvidenceSource.CORRELATION,
                    confidence=0.99,
                    distance=0,
                    detail="stuck",
                ),
            )
        )
        supervisor = AdaptiveScrollSupervisor(
            device=device,
            detector=detector,
            surface=FakeSurface(hints=()),
        )
        converter = CoordinateConverter(logical_width=1080, logical_height=2340)
        region = converter.region_from_bounds(
            bounds=Bounds(
                x=0,
                y=393,
                width=1080,
                height=1632,
                coordinate_system=CoordinateSystem.DEVICE_PIXEL,
            ),
            source=CoordinateSource.MODEL,
        )
        current = converter.resolve_scroll_path(region=region, direction="down")

        await supervisor.execute(
            before=before,
            observation=self.__empty_observation(),
            context=ScrollContext(direction=ScrollDirection.DOWN, region=region),
            current=current,
            converter=converter,
            policy=DeviceRuntimeConfiguration().interaction.policy.scroll.adaptive.model_copy(
                update={
                    "enabled": True,
                    "maximum_attempts": 1,
                    "suspicious_bottom_ratio": 1.0,
                }
            ),
        )

        self.assertEqual(len(device.swipes), 1)
        self.assertGreaterEqual(abs(device.swipes[0][1] - device.swipes[0][3]), 1200)

    def test_rejects_corridors_that_shrink_scroll_below_minimum_travel_ratio(self) -> None:
        """
        Planner must keep the original long path when only micro-corridors are available.
        """

        planner = AdaptiveScrollPlanner()
        converter = CoordinateConverter(logical_width=1080, logical_height=2340)
        region = converter.region_from_bounds(
            bounds=Bounds(
                x=0,
                y=393,
                width=1080,
                height=1861,
                coordinate_system=CoordinateSystem.DEVICE_PIXEL,
            ),
            source=CoordinateSource.MODEL,
        )
        current = converter.resolve_scroll_path(region=region, direction="down")
        attempts = planner.plan(
            context=ScrollContext(direction=ScrollDirection.DOWN, region=region),
            current=current,
            scope=self.__scope(region=region),
            surfaces=(),
            converter=converter,
            policy=DeviceRuntimeConfiguration().interaction.policy.scroll.adaptive.model_copy(
                update={"enabled": True, "maximum_attempts": 3, "suspicious_bottom_ratio": 1.0}
            ),
            capture_height=2340,
        )

        self.assertGreaterEqual(attempts[0].path.distance, int(current.distance * 0.70))

    def test_keeps_centered_scope_path_even_when_content_fills_the_feed(self) -> None:
        """
        Feed content inside the resolved container must not prevent planning a real swipe.
        """

        planner = AdaptiveScrollPlanner()
        converter = CoordinateConverter(logical_width=1080, logical_height=2340)
        region = converter.region_from_bounds(
            bounds=Bounds(
                x=0,
                y=393,
                width=1080,
                height=1861,
                coordinate_system=CoordinateSystem.DEVICE_PIXEL,
            ),
            source=CoordinateSource.MODEL,
        )
        current = converter.resolve_scroll_path(region=region, direction="down")
        attempts = planner.plan(
            context=ScrollContext(direction=ScrollDirection.DOWN, region=region),
            current=current,
            scope=self.__scope(region=region),
            surfaces=(),
            converter=converter,
            policy=DeviceRuntimeConfiguration().interaction.policy.scroll.adaptive.model_copy(
                update={"enabled": True, "maximum_attempts": 3, "suspicious_bottom_ratio": 1.0}
            ),
            capture_height=2340,
        )

        self.assertGreaterEqual(len(attempts), 1)
        self.assertGreaterEqual(attempts[0].path.distance, 260)
        self.assertEqual(attempts[0].path.start_x, attempts[0].path.end_x)

    async def test_stops_when_scroll_budget_is_exhausted(self) -> None:
        """
        Stop adaptive retries once the configured wall-time budget is spent.
        """

        before = ScreenCapture(
            width=1080,
            height=2340,
            activity="com.test.app",
            image=b"before",
            timestamp=1,
        )
        device = FakeDevice(after_sequence=(b"after_one", b"after_two"))
        detector = FakeDetect(
            verdicts=(
                ScrollVerdict(
                    kind=ScrollVerdictKind.NO_PROGRESS,
                    source=ScrollEvidenceSource.CORRELATION,
                    confidence=0.99,
                    distance=0,
                    detail="no_move",
                ),
                ScrollVerdict(
                    kind=ScrollVerdictKind.NO_PROGRESS,
                    source=ScrollEvidenceSource.CORRELATION,
                    confidence=0.99,
                    distance=0,
                    detail="still_no_move",
                ),
            )
        )
        supervisor = AdaptiveScrollSupervisor(
            device=device,
            detector=detector,
            surface=FakeSurface(hints=()),
        )
        converter = CoordinateConverter(logical_width=1080, logical_height=2340)
        region = converter.viewport_region()
        current = converter.resolve_scroll_path(region=region, direction="down")

        with patch(
            "fathom.core.execution.command.supervisor.time.time",
            side_effect=(100.0, 100.0, 100.020, 100.020, 100.030, 100.030),
        ):
            result, outcome, _ = await supervisor.execute(
                before=before,
                observation=self.__observation(),
                context=ScrollContext(direction=ScrollDirection.DOWN, region=region),
                current=current,
                converter=converter,
                policy=DeviceRuntimeConfiguration().interaction.policy.scroll.adaptive.model_copy(
                    update={"enabled": True, "maximum_attempts": 3, "budget": 10}
                ),
            )

        self.assertFalse(result.success)
        self.assertEqual(len(device.swipes), 1)
        self.assertEqual(len(outcome.attempts), 1)

    @staticmethod
    def __capture(*, name: str) -> ScreenCapture:
        """
        Load one real Healthkart screenshot fixture.
        """

        path = PROJECT_ROOT / "tests/fixtures/execution/android/scroll/001" / name
        return ScreenCapture(
            width=1080,
            height=2340,
            activity="com.aranoah.healthkart.plus",
            image=path.read_bytes(),
            timestamp=0,
        )

    @staticmethod
    def __observation() -> ScreenObservation:
        """
        Build the minimal observation needed by the supervisor.
        """

        return ScreenObservation(
            activity="com.aranoah.healthkart.plus",
            hashes=ScreenHashBundle(visual_hash="a", xml_hash="b", interaction_hash="c"),
            elements=(
                PerceivedElement(
                    identifier="coupon",
                    bounds=Bounds(
                        x=128,
                        y=1881,
                        width=824,
                        height=62,
                        coordinate_system=CoordinateSystem.DEVICE_PIXEL,
                    ),
                    source=ElementSource.XML,
                    role=ElementRole.TEXT,
                    confidence=1.0,
                    text="Apply coupon to get EXTRA 15% OFF",
                    tappable=True,
                ),
            ),
            overlays=(),
            keyboard=KeyboardObservation(visible=False, bounds=None, dismiss=()),
            scroll=(),
            calls_to_action=(),
            focused=None,
        )

    @staticmethod
    def __scope(*, region: ExecutionRegion) -> ScrollScope:
        """
        Build a resolved scroll scope for planner-only tests.
        """

        return ScrollScope(
            identifier="scope",
            kind=CommandScopeKind.VIEWPORT,
            bounds=Bounds(
                x=region.x,
                y=region.y,
                width=region.width,
                height=region.height,
                coordinate_system=CoordinateSystem.DEVICE_PIXEL,
            ),
            region=region,
            axis="vertical",
            confidence=1.0,
            source=ScrollEvidenceSource.SURFACE,
            label=None,
        )

    @staticmethod
    def __empty_observation() -> ScreenObservation:
        """
        Build an observation with no blockers inside the corridor.
        """

        return ScreenObservation(
            activity="com.test.app",
            hashes=ScreenHashBundle(visual_hash="a", xml_hash="b", interaction_hash="c"),
            elements=(),
            overlays=(),
            keyboard=KeyboardObservation(visible=False, bounds=None, dismiss=()),
            scroll=(),
            calls_to_action=(),
            focused=None,
        )

    @staticmethod
    def __occupied_observation() -> ScreenObservation:
        """
        Build an observation where the preferred start point sits inside a content card.
        """

        return ScreenObservation(
            activity="com.test.app",
            hashes=ScreenHashBundle(visual_hash="a", xml_hash="b", interaction_hash="c"),
            elements=(
                PerceivedElement(
                    identifier="card",
                    bounds=Bounds(
                        x=360,
                        y=1650,
                        width=360,
                        height=180,
                        coordinate_system=CoordinateSystem.DEVICE_PIXEL,
                    ),
                    source=ElementSource.XML,
                    role=ElementRole.BUTTON,
                    confidence=1.0,
                    text="Featured card",
                    tappable=True,
                ),
            ),
            overlays=(),
            keyboard=KeyboardObservation(visible=False, bounds=None, dismiss=()),
            scroll=(),
            calls_to_action=(),
            focused=None,
        )


class FakeDetect(ScrollDetectPort):
    """
    Deterministic detector double.
    """

    def __init__(self, *, verdicts: Tuple[ScrollVerdict, ...]) -> None:
        self.__verdicts = list(verdicts)

    async def evaluate(
        self,
        *,
        before: ScreenCapture,
        after: ScreenCapture,
        region: Bounds,
        direction: ScrollDirection,
    ) -> ScrollVerdict:
        _ = before, after, region, direction
        return self.__verdicts.pop(0)


class RecordingDetect(ScrollDetectPort):
    """
    Detector double that records the before image used for each retry evaluation.
    """

    def __init__(self, *, verdicts: Tuple[ScrollVerdict, ...]) -> None:
        self.__verdicts = list(verdicts)
        self.before_images = []
        self.after_activities = []
        self.after_xml = []

    async def evaluate(
        self,
        *,
        before: ScreenCapture,
        after: ScreenCapture,
        region: Bounds,
        direction: ScrollDirection,
    ) -> ScrollVerdict:
        _ = region, direction
        self.before_images.append(before.image)
        self.after_activities.append(after.activity)
        self.after_xml.append(after.xml_content)
        return self.__verdicts.pop(0)


class FakeSurface(ScrollSurfacePort):
    """
    Surface inspector double.
    """

    def __init__(self, *, hints: Tuple[ScrollSurface, ...]) -> None:
        self.hints = hints

    async def inspect(
        self,
        *,
        observation: ScreenObservation,
        path: GesturePath,
        capture_width: int,
        capture_height: int,
    ) -> Tuple[ScrollSurface, ...]:
        _ = observation, path, capture_width, capture_height
        return self.hints


class FakeDevice(DevicePort):
    """
    Device double that records swipe attempts.
    """

    def __init__(
        self,
        *,
        after: bytes | None = None,
        after_sequence: Tuple[bytes, ...] = (),
    ) -> None:
        self.swipes = []
        self.__after = after or b"after"
        self.__after_sequence = list(after_sequence)

    @property
    def configuration(self) -> Optional[DeviceRuntimeConfiguration]:
        return DeviceRuntimeConfiguration()

    async def tap(self, *, x: int, y: int) -> ActionResult:
        raise AssertionError("tap not expected")

    async def type(
        self,
        *,
        text: str,
        prefilled: str = "",
        replace: bool = True,
        locator: Optional[str] = None,
    ) -> ActionResult:
        raise AssertionError("type not expected")

    async def swipe(
        self,
        *,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        duration: Optional[int] = None,
        speed: Optional[SwipeSpeed] = None,
    ) -> ActionResult:
        self.swipes.append((x1, y1, x2, y2, duration, speed))
        return ActionResult(success=True, duration=1)

    async def back(self) -> ActionResult:
        raise AssertionError("back not expected")

    async def home(self) -> ActionResult:
        raise AssertionError("home not expected")

    async def get_current_package(self) -> str:
        return "com.aranoah.healthkart.plus"

    async def capture_screen(self) -> bytes:
        if self.__after_sequence:
            return self.__after_sequence.pop(0)
        return self.__after

    async def dump_hierarchy(self) -> Optional[str]:
        return None

    async def get_snapshot(self) -> Tuple[bytes, Optional[str]]:
        return await self.capture_screen(), None

    async def get_dimensions(self) -> Tuple[int, int]:
        return 1080, 2340

    async def wait_for_device(self, *, timeout: float) -> bool:
        _ = timeout
        return True
