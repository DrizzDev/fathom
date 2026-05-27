from __future__ import annotations

import unittest
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, Mock, patch

from fathom.constants import ActionType
from fathom.constants.interaction import SwipeSpeed
from fathom.constants.observation import KeyboardVisibility
from fathom.core.services.action import ActionExecutor
from fathom.interfaces.device import DevicePort
from fathom.schemas.actions import (
    Action,
    Bounds,
    CoordinateSystem,
    InputContext,
    InputContextSource,
)
from fathom.schemas.configuration import (
    DeviceRuntimeConfiguration,
    InteractionPolicyConfiguration,
    InteractionRuntimeConfiguration,
    TypeInteractionPolicy,
)
from fathom.schemas.observation import KeyboardObservation, ScreenObservation
from fathom.schemas.results import ActionResult
from fathom.schemas.screens import ScreenCapture, ScreenHashBundle
from fathom.schemas.steps import Step

# Use a near-zero focus delay for fast tests.
FAST_TYPE_POLICY = DeviceRuntimeConfiguration(
    interaction=InteractionRuntimeConfiguration(
        policy=InteractionPolicyConfiguration(
            type=TypeInteractionPolicy(delay=10),
        ),
    ),
)

SWIPE_RETRY_POLICY = FAST_TYPE_POLICY

MINIMAL_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc```\x00\x00"
    b"\x00\x04\x00\x01\xf6\x178U\x00\x00\x00\x00IEND\xaeB`\x82"
)


class FakeDevice(DevicePort):
    """
    In-memory device double that records all calls for assertion.
    """

    def __init__(
        self,
        *,
        dimensions: Tuple[int, int] = (1080, 2340),
        type_results: Optional[List[ActionResult]] = None,
        swipe_results: Optional[List[ActionResult]] = None,
        device_configuration: Optional[DeviceRuntimeConfiguration] = None,
    ) -> None:
        """
        Initialize fake device with configurable type results.
        """

        self.__tap_calls: List[Tuple[int, int]] = []
        self.__type_calls: List[Dict[str, Any]] = []
        self.__swipe_calls: List[Dict[str, Any]] = []
        self.__enter_calls = 0

        self.__dimensions = dimensions
        self.__configuration = device_configuration or FAST_TYPE_POLICY
        self.__type_results = list(type_results or [ActionResult(success=True, duration=1)])
        self.__swipe_results = list(swipe_results or [ActionResult(success=True, duration=1)])

    @property
    def tap_calls(self) -> List[Tuple[int, int]]:
        """
        Recorded tap coordinates.
        """

        return self.__tap_calls

    @property
    def type_calls(self) -> List[Dict[str, Any]]:
        """
        Recorded type call parameters.
        """

        return self.__type_calls

    @property
    def swipe_calls(self) -> List[Dict[str, Any]]:
        """
        Recorded swipe call parameters.
        """

        return self.__swipe_calls

    @property
    def enter_calls(self) -> int:
        """
        Recorded enter invocations.
        """

        return self.__enter_calls

    @property
    def configuration(self) -> Optional[DeviceRuntimeConfiguration]:
        """
        Return device runtime configuration.
        """

        return self.__configuration

    async def tap(self, *, x: int, y: int) -> ActionResult:
        """
        Record tap coordinates and return success.
        """

        self.__tap_calls.append((x, y))
        return ActionResult(success=True, duration=1)

    async def type(
        self,
        *,
        text: str,
        prefilled: str = "",
        replace: bool = True,
        locator: Optional[str] = None,
    ) -> ActionResult:
        """
        Record type parameters and return the next queued result.
        """

        self.__type_calls.append(
            {"text": text, "locator": locator, "replace": replace, "prefilled": prefilled}
        )
        return (
            self.__type_results.pop(0)
            if self.__type_results
            else ActionResult(success=True, duration=1)
        )

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
        """
        Return success for swipe actions.
        """

        self.__swipe_calls.append(
            {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "duration": duration,
                "speed": speed,
            }
        )
        return (
            self.__swipe_results.pop(0)
            if self.__swipe_results
            else ActionResult(success=True, duration=1)
        )

    async def back(self) -> ActionResult:
        """
        Return success for back action.
        """

        return ActionResult(success=True, duration=1)

    async def home(self) -> ActionResult:
        """
        Return success for home action.
        """

        return ActionResult(success=True, duration=1)

    async def enter(self) -> ActionResult:
        """
        Record enter and return success.
        """

        self.__enter_calls += 1
        return ActionResult(success=True, duration=1)

    async def get_current_package(self) -> str:
        """
        Return a stable package name.
        """

        return "com.test.app"

    async def capture_screen(self) -> bytes:
        """
        Return minimal PNG bytes.
        """

        return MINIMAL_PNG

    async def dump_hierarchy(self) -> Optional[str]:
        """
        Return no hierarchy.
        """

        return None

    async def get_snapshot(self) -> Tuple[bytes, Optional[str]]:
        """
        Return screenshot-only snapshot.
        """

        return await self.capture_screen(), None

    async def get_dimensions(self) -> Tuple[int, int]:
        """
        Return configured dimensions.
        """

        return self.__dimensions

    async def wait_for_device(self, *, timeout: float) -> bool:
        """
        Always report ready.
        """

        return True


class ActionExecutorTypeTest(unittest.IsolatedAsyncioTestCase):
    """
    Cover the TYPE action execution path including resource-id forwarding,
    replace semantics, and retry-on-failure behavior.
    """

    @staticmethod
    def __build_action(
        *,
        text: str = "hello",
        label_id: Optional[str] = None,
        bounds: Optional[Bounds] = None,
        input_context: Optional[InputContext] = None,
    ) -> Action:
        """
        Build a minimal TYPE action with pixel bounds for testing.
        """

        return Action(
            text=text,
            rationale="test",
            label_id=label_id,
            input_context=input_context,
            action_type=ActionType.TYPE,
            bounds=bounds
            or Bounds(
                x=100,
                y=200,
                width=400,
                height=100,
                coordinate_system=CoordinateSystem.DEVICE_PIXEL,
            ),
        )

    @staticmethod
    def __build_step(action: Action) -> Step:
        """
        Wrap an action in a minimal Step for the executor.
        """

        return Step(
            metadata={},
            action=action,
            step_number=1,
            condition=None,
            event_type="action",
            screen_hash="abc123",
            is_conditional=False,
        )

    @staticmethod
    def __build_executor(
        device: FakeDevice,
        *,
        pipeline: Optional[Mock] = None,
    ) -> ActionExecutor:
        """
        Build an ActionExecutor with minimal dependencies.
        """

        return ActionExecutor(
            max_retries=0,
            device=device,
            telemetry=Mock(),
            path_manager=Mock(),
            pipeline=pipeline,
        )

    @staticmethod
    def __build_capture() -> ScreenCapture:
        """
        Build a minimal ScreenCapture for the executor.
        """

        return ScreenCapture(
            width=1080,
            height=2340,
            timestamp=0,
            image=b"",
            activity="com.test.app",
        )

    async def test_type_forwards_locator_from_input_context(self) -> None:
        """
        Locator from input_context is forwarded to device.type().
        """

        device = FakeDevice()
        executor = self.__build_executor(device)

        action = self.__build_action(
            text="chennai adyar",
            input_context=InputContext(
                locator="com.app:id/searchField",
                source=InputContextSource.XML,
            ),
        )

        result = await executor.act(
            session_id="session__1",
            package_name="com.test.app",
            step=self.__build_step(action),
            pre_capture=self.__build_capture(),
        )

        self.assertTrue(result.success)
        self.assertEqual(len(device.type_calls), 1)
        self.assertEqual(device.type_calls[0]["text"], "chennai adyar")
        self.assertEqual(device.type_calls[0]["locator"], "com.app:id/searchField")

    async def test_enter_executes_device_enter_without_coordinate_fallback(self) -> None:
        """
        ENTER must dispatch the keyboard primitive instead of becoming WAIT.
        """

        device = FakeDevice()
        executor = self.__build_executor(device)
        action = Action(
            action_type=ActionType.ENTER,
            target="Search button on keyboard",
            rationale="submit search",
            confidence=0.9,
        )

        result = await executor.act(
            session_id="session__1",
            package_name="com.test.app",
            step=self.__build_step(action),
            pre_capture=self.__build_capture(),
        )

        self.assertTrue(result.success)
        self.assertEqual(device.enter_calls, 1)
        self.assertEqual(device.tap_calls, [])

    async def test_type_sets_replace_when_prefilled_present(self) -> None:
        """
        When input_context carries prefilled text, replace=True and prefilled
        are forwarded so the provider clears before typing.
        """

        device = FakeDevice()
        executor = self.__build_executor(device)

        action = self.__build_action(
            text="new value",
            input_context=InputContext(locator="com.app:id/input", prefilled="old value"),
        )

        result = await executor.act(
            session_id="session__1",
            package_name="com.test.app",
            step=self.__build_step(action),
            pre_capture=self.__build_capture(),
        )

        self.assertTrue(result.success)
        self.assertEqual(device.type_calls[0]["replace"], True)
        self.assertEqual(device.type_calls[0]["prefilled"], "old value")

    async def test_type_skips_replace_when_field_is_empty(self) -> None:
        """
        When input_context has no prefilled text, replace is False.
        """

        device = FakeDevice()
        executor = self.__build_executor(device)

        action = self.__build_action(
            text="fresh text",
            input_context=InputContext(locator="com.app:id/field"),
        )

        result = await executor.act(
            session_id="session__1",
            package_name="com.test.app",
            step=self.__build_step(action),
            pre_capture=self.__build_capture(),
        )

        self.assertTrue(result.success)
        self.assertEqual(device.type_calls[0]["replace"], False)
        self.assertEqual(device.type_calls[0]["prefilled"], "")

    async def test_type_retries_with_retap_on_failure(self) -> None:
        """
        When the first type attempt fails, the executor re-taps for focus,
        waits, and retries once.
        """

        device = FakeDevice(
            type_results=[
                ActionResult(success=False, error="No active element found", duration=1),
                ActionResult(success=True, duration=1),
            ]
        )
        executor = self.__build_executor(device)

        action = self.__build_action(text="retry text")

        result = await executor.act(
            session_id="session__1",
            package_name="com.test.app",
            step=self.__build_step(action),
            pre_capture=self.__build_capture(),
        )

        self.assertTrue(result.success)
        self.assertEqual(len(device.type_calls), 2)
        # Initial tap + re-tap before retry
        self.assertEqual(len(device.tap_calls), 2)

    async def test_type_returns_failure_when_retry_also_fails(self) -> None:
        """
        When both the initial attempt and the retry fail, the executor returns
        the failure result.
        """

        device = FakeDevice(
            type_results=[
                ActionResult(success=False, error="No active element found", duration=1),
                ActionResult(success=False, error="Still no active element", duration=1),
            ]
        )
        executor = self.__build_executor(device)

        action = self.__build_action(text="will fail")

        result = await executor.act(
            session_id="session__1",
            package_name="com.test.app",
            step=self.__build_step(action),
            pre_capture=self.__build_capture(),
        )

        self.assertFalse(result.success)
        self.assertEqual(len(device.tap_calls), 2)
        self.assertEqual(len(device.type_calls), 2)

    async def test_type_works_without_input_context(self) -> None:
        """
        When no input_context is attached (no XML grounding), type proceeds
        with no locator and no replace.
        """

        device = FakeDevice()
        executor = self.__build_executor(device)

        action = self.__build_action(text="no grounding")

        result = await executor.act(
            session_id="session__1",
            package_name="com.test.app",
            step=self.__build_step(action),
            pre_capture=self.__build_capture(),
        )

        self.assertTrue(result.success)
        self.assertEqual(device.type_calls[0]["locator"], None)
        self.assertEqual(device.type_calls[0]["prefilled"], "")
        self.assertEqual(device.type_calls[0]["replace"], False)

    async def test_type_taps_before_typing(self) -> None:
        """
        The executor taps at the element center to focus before typing.
        """

        device = FakeDevice()
        executor = self.__build_executor(device)

        action = self.__build_action(
            text="tap first",
            bounds=Bounds(
                x=100,
                y=200,
                width=400,
                height=100,
                coordinate_system=CoordinateSystem.DEVICE_PIXEL,
            ),
        )

        await executor.act(
            session_id="session__1",
            package_name="com.test.app",
            step=self.__build_step(action),
            pre_capture=self.__build_capture(),
        )

        self.assertEqual(len(device.tap_calls), 1)
        # y=250 shifted up by 20% tap bias (250 - 20 = 230)
        self.assertEqual(device.tap_calls[0], (300, 230))
        self.assertEqual(len(device.type_calls), 1)

    @patch("fathom.core.services.action.asyncio.sleep", new_callable=AsyncMock)
    async def test_type_waits_after_focus_tap_before_typing(self, mock_sleep: AsyncMock) -> None:
        """
        A stabilisation wait occurs after the focus tap and before the first
        type call. This is the root fix for the LambdaTest active-element race.
        """

        device = FakeDevice()
        executor = self.__build_executor(device)

        action = self.__build_action(text="wait test")

        await executor.act(
            session_id="session__1",
            package_name="com.test.app",
            step=self.__build_step(action),
            pre_capture=self.__build_capture(),
        )

        mock_sleep.assert_awaited_once()
        self.assertEqual(len(device.tap_calls), 1)
        self.assertEqual(len(device.type_calls), 1)

    async def test_non_coordinate_action_emits_no_trace_record(self) -> None:
        """
        Non-gesture actions must not emit synthetic trace artifacts.
        """

        device = FakeDevice()
        pipeline = Mock()
        pipeline.emit = AsyncMock()
        executor = self.__build_executor(device, pipeline=pipeline)

        action = Action(
            action_type=ActionType.HIDE_KEYBOARD,
            target="keyboard",
            rationale="dismiss keyboard",
            confidence=1.0,
        )

        result = await executor.act(
            session_id="session__1",
            package_name="com.test.app",
            step=self.__build_step(action),
            pre_capture=self.__build_capture(),
        )

        self.assertTrue(result.success)
        pipeline.emit.assert_not_awaited()

    async def test_tap_emits_trace_record_with_real_coordinates(self) -> None:
        """
        Coordinate-backed device taps should emit one real trace artifact.
        """

        device = FakeDevice()
        pipeline = Mock()
        pipeline.emit = AsyncMock()
        executor = self.__build_executor(device, pipeline=pipeline)

        action = Action(
            action_type=ActionType.TAP,
            target="search box",
            rationale="focus input",
            confidence=1.0,
            bounds=Bounds(
                x=100,
                y=200,
                width=300,
                height=120,
                coordinate_system=CoordinateSystem.DEVICE_PIXEL,
            ),
        )

        result = await executor.act(
            session_id="session__1",
            package_name="com.test.app",
            step=self.__build_step(action),
            pre_capture=self.__build_capture(),
        )

        self.assertTrue(result.success)
        pipeline.emit.assert_awaited_once()
        record = pipeline.emit.await_args.kwargs["record"]
        self.assertNotEqual(record.payload.coords, ())

    @patch("fathom.core.services.action.asyncio.sleep", new_callable=AsyncMock)
    async def test_type_retry_waits_twice(self, mock_sleep: AsyncMock) -> None:
        """
        On retry, a second stabilisation wait occurs (one per focus-and-type attempt).
        """

        device = FakeDevice(
            type_results=[
                ActionResult(success=False, error="No active element", duration=1),
                ActionResult(success=True, duration=1),
            ]
        )
        executor = self.__build_executor(device)

        action = self.__build_action(text="retry wait")

        await executor.act(
            session_id="session__1",
            package_name="com.test.app",
            step=self.__build_step(action),
            pre_capture=self.__build_capture(),
        )

        self.assertEqual(mock_sleep.await_count, 2)
        self.assertEqual(len(device.tap_calls), 2)
        self.assertEqual(len(device.type_calls), 2)


class ActionExecutorScrollRetryTest(unittest.IsolatedAsyncioTestCase):
    """
    Pin scroll-specific retry behavior in :class:`ActionExecutor`.
    """

    @staticmethod
    def __build_action() -> Action:
        """
        Build a minimal swipe action for executor tests.
        """

        return Action(
            action_type=ActionType.SWIPE_UP,
            target="Scroll list",
            rationale="test swipe",
            confidence=1.0,
            bounds=Bounds(
                x=120,
                y=600,
                width=600,
                height=1200,
                coordinate_system=CoordinateSystem.DEVICE_PIXEL,
            ),
        )

    @classmethod
    def __build_step(cls) -> Step:
        """
        Wrap the swipe action in a step fixture.
        """

        return Step(
            metadata={},
            action=cls.__build_action(),
            step_number=1,
            condition=None,
            event_type="action",
            screen_hash="scroll123",
            is_conditional=False,
        )

    @staticmethod
    def __build_capture() -> ScreenCapture:
        """
        Build a minimal capture fixture.
        """

        return ScreenCapture(
            width=1080,
            height=2340,
            timestamp=0,
            image=MINIMAL_PNG,
            activity="com.test.app",
        )

    @staticmethod
    def __build_observation(*, keyboard: KeyboardVisibility) -> ScreenObservation:
        """
        Build a minimal observation with the requested keyboard state.
        """

        return ScreenObservation(
            activity="com.test.app",
            elements=(),
            hashes=ScreenHashBundle(
                visual_hash="0" * 16,
                xml_hash="a" * 16,
                interaction_hash="b" * 16,
            ),
            keyboard=KeyboardObservation(visibility=keyboard),
            overlays=(),
            scroll=(),
            calls_to_action=(),
        )

    async def test_swipe_failure_does_not_use_outer_retry_loop(self) -> None:
        """
        Outer ``act()`` must not multiply swipe attempts; only the inner coordinator retries.
        """

        device = FakeDevice(
            swipe_results=[
                ActionResult(success=False, error="gesture blocked", duration=1) for _ in range(20)
            ],
            device_configuration=SWIPE_RETRY_POLICY,
        )
        executor = ActionExecutor(
            max_retries=2,
            device=device,
            telemetry=Mock(),
            path_manager=Mock(),
        )

        result = await executor.act(
            session_id="session__1",
            package_name="com.test.app",
            step=self.__build_step(),
            pre_capture=self.__build_capture(),
            observation=ScreenObservation(
                activity="com.test.app",
                elements=(),
                hashes=ScreenHashBundle(
                    visual_hash="0" * 16,
                    xml_hash="a" * 16,
                    interaction_hash="b" * 16,
                ),
                keyboard=KeyboardObservation(visibility=KeyboardVisibility.HIDDEN),
                overlays=(),
                scroll=(),
                calls_to_action=(),
            ),
        )

        self.assertFalse(result.success)
        # Inner coordinator may dispatch up to 1 + len(magnitudes) attempts; the outer
        # ``max_retries`` loop must not multiply that count.
        self.assertLessEqual(len(device.swipe_calls), 4)

    async def test_system_scroll_is_blocked_when_keyboard_is_visible(self) -> None:
        """
        Mechanical SCROLL recovery must not dispatch into a visible keyboard surface.
        """

        device = FakeDevice(device_configuration=SWIPE_RETRY_POLICY)
        executor = ActionExecutor(
            max_retries=0,
            device=device,
            telemetry=Mock(),
            path_manager=Mock(),
        )
        step = Step(
            metadata={},
            step_number=6,
            condition=None,
            event_type="action",
            screen_hash="otp123",
            is_conditional=False,
            action=Action(
                action_type=ActionType.SCROLL,
                target="system: scroll",
                rationale="recovery",
                confidence=1.0,
            ),
        )

        result = await executor.act(
            session_id="keyboard-visible-scroll-regression",
            package_name="com.test.app",
            step=step,
            pre_capture=self.__build_capture(),
            observation=self.__build_observation(keyboard=KeyboardVisibility.VISIBLE),
        )

        self.assertFalse(result.success)
        self.assertEqual(device.swipe_calls, [])
        self.assertEqual(result.error, "scroll blocked by visible keyboard")


class _DeviceWithoutBack(FakeDevice):
    """
    FakeDevice variant whose back() raises NotImplementedError, modelling
    iOS adapters that have no system back gesture.
    """

    async def back(self) -> ActionResult:
        """
        Refuse back navigation the way an iOS adapter does.
        """

        raise NotImplementedError("iOS has no system-level back gesture.")


class ActionExecutorBackUnsupportedTest(unittest.IsolatedAsyncioTestCase):
    """
    Verify the executor returns a graceful soft-failure ActionResult instead
    of letting NotImplementedError propagate when a device refuses BACK.
    """

    @staticmethod
    def __build_step(action: Action) -> Step:
        """
        Wrap an action in a minimal Step for the executor.
        """

        return Step(
            metadata={},
            action=action,
            step_number=1,
            condition=None,
            event_type="action",
            screen_hash="abc123",
            is_conditional=False,
        )

    @staticmethod
    def __build_executor(device: DevicePort) -> ActionExecutor:
        """
        Build a minimal ActionExecutor bound to the supplied device.
        """

        return ActionExecutor(
            max_retries=0,
            device=device,
            telemetry=Mock(),
            path_manager=Mock(),
            pipeline=None,
        )

    @staticmethod
    def __build_capture() -> ScreenCapture:
        """
        Build a minimal ScreenCapture.
        """

        return ScreenCapture(
            width=1080,
            height=2340,
            timestamp=0,
            image=b"",
            activity="com.test.app",
        )

    async def test_back_returns_soft_failure_when_device_unsupported(self) -> None:
        """
        ActionType.BACK against an iOS-style adapter returns success=False
        with a descriptive error rather than raising into the executor.
        """

        device = _DeviceWithoutBack()
        executor = self.__build_executor(device)
        action = Action(
            action_type=ActionType.BACK,
            target="system: back",
            rationale="loop recovery",
            confidence=0.9,
        )

        result = await executor.act(
            session_id="session__1",
            package_name="com.test.app",
            step=self.__build_step(action),
            pre_capture=self.__build_capture(),
        )

        self.assertFalse(result.success)
        self.assertIn("does not support back", str(result.error or ""))

    async def test_hide_keyboard_back_fallback_returns_soft_failure(self) -> None:
        """
        ActionType.HIDE_KEYBOARD on a device without native hide_keyboard
        falls back to back(); when back() also raises NotImplementedError,
        the executor returns a soft-failure ActionResult rather than crashing.
        """

        device = _DeviceWithoutBack()
        executor = self.__build_executor(device)
        action = Action(
            action_type=ActionType.HIDE_KEYBOARD,
            target="keyboard dismiss",
            rationale="dismiss soft keyboard",
            confidence=0.9,
        )

        result = await executor.act(
            session_id="session__1",
            package_name="com.test.app",
            step=self.__build_step(action),
            pre_capture=self.__build_capture(),
        )

        self.assertFalse(result.success)
        self.assertIn("hide keyboard", (result.error or "").lower())
