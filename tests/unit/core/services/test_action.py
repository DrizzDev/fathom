from __future__ import annotations

import unittest
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, Mock, patch

from fathom.constants import ActionType
from fathom.constants.interaction import SwipeSpeed
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
from fathom.schemas.results import ActionResult
from fathom.schemas.screens import ScreenCapture
from fathom.schemas.steps import Step

# Use a near-zero focus delay for fast tests.
FAST_TYPE_POLICY = DeviceRuntimeConfiguration(
    interaction=InteractionRuntimeConfiguration(
        policy=InteractionPolicyConfiguration(
            type=TypeInteractionPolicy(delay=10),
        ),
    ),
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
        device_configuration: Optional[DeviceRuntimeConfiguration] = None,
    ) -> None:
        """
        Initialize fake device with configurable type results.
        """

        self.__tap_calls: List[Tuple[int, int]] = []
        self.__type_calls: List[Dict[str, Any]] = []

        self.__dimensions = dimensions
        self.__configuration = device_configuration or FAST_TYPE_POLICY
        self.__type_results = list(type_results or [ActionResult(success=True, duration=1)])

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

        return ActionResult(success=True, duration=1)

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

    async def get_current_package(self) -> str:
        """
        Return a stable package name.
        """

        return "com.test.app"

    async def capture_screen(self) -> bytes:
        """
        Return minimal PNG bytes.
        """

        return b"\x89PNG\r\n\x1a\n" + b"\x00" * 16

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
    def __build_executor(device: FakeDevice) -> ActionExecutor:
        """
        Build an ActionExecutor with minimal dependencies.
        """

        return ActionExecutor(
            max_retries=0,
            device=device,
            telemetry=Mock(),
            path_manager=Mock(),
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
            image=b"fake",
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
