from __future__ import annotations

import unittest
from unittest.mock import AsyncMock

from fathom.adapters.swipe import DeviceSwipeDispatcher
from fathom.interfaces.vision import VisualHasher
from fathom.schemas.actions import GesturePath
from fathom.schemas.results import ActionResult


class _FixedHasher(VisualHasher):
    """
    Test hasher returning a fixed digest regardless of input bytes.
    """

    def __init__(self, *, value: str) -> None:
        self.__value = value
        self.calls: int = 0

    def hash(self, *, image: bytes) -> str:
        _ = image
        self.calls += 1
        return self.__value


class DeviceSwipeDispatcherTest(unittest.IsolatedAsyncioTestCase):
    """
    Cover dispatch outcomes for the concrete device-backed swipe dispatcher.
    """

    @staticmethod
    def __path() -> GesturePath:
        """
        Build one upward swipe gesture.
        """

        return GesturePath(start_x=540, start_y=1900, end_x=540, end_y=600, duration=300)

    @staticmethod
    def __device(*, swipe_result: ActionResult, capture_bytes: bytes = b"img") -> AsyncMock:
        """
        Build a mocked DevicePort returning the supplied swipe result and capture bytes.
        """

        mock = AsyncMock()
        mock.swipe = AsyncMock(return_value=swipe_result)
        mock.capture_screen = AsyncMock(return_value=capture_bytes)
        return mock

    async def test_visual_change_when_hashes_differ(self) -> None:
        """
        When device swipe succeeds and post hash differs from before hash, the attempt is changed.
        """

        device = self.__device(swipe_result=ActionResult(success=True, duration=1))
        hasher = _FixedHasher(value="POST")
        dispatcher = DeviceSwipeDispatcher(device=device, hasher=hasher)

        attempt = await dispatcher.attempt(path=self.__path(), index=0, original_before="PRE")

        self.assertTrue(attempt.device.succeeded)
        self.assertTrue(attempt.visual.changed)
        self.assertEqual(attempt.visual.before, "PRE")
        self.assertEqual(attempt.visual.after, "POST")
        self.assertEqual(hasher.calls, 1)

    async def test_no_visual_change_when_hashes_equal(self) -> None:
        """
        When the post-attempt hash equals the original-before hash, visual.changed is False.
        """

        device = self.__device(swipe_result=ActionResult(success=True, duration=1))
        hasher = _FixedHasher(value="SAME")
        dispatcher = DeviceSwipeDispatcher(device=device, hasher=hasher)

        attempt = await dispatcher.attempt(path=self.__path(), index=0, original_before="SAME")

        self.assertTrue(attempt.device.succeeded)
        self.assertFalse(attempt.visual.changed)

    async def test_device_failure_skips_capture(self) -> None:
        """
        On device failure, no post-capture or hash is performed and visual.after is None.
        """

        device = self.__device(
            swipe_result=ActionResult(success=False, duration=1, error="gesture rejected"),
        )
        hasher = _FixedHasher(value="POST")
        dispatcher = DeviceSwipeDispatcher(device=device, hasher=hasher)

        attempt = await dispatcher.attempt(path=self.__path(), index=2, original_before="PRE")

        self.assertFalse(attempt.device.succeeded)
        self.assertEqual(attempt.device.error, "gesture rejected")
        self.assertIsNone(attempt.visual.after)
        self.assertFalse(attempt.visual.changed)
        device.capture_screen.assert_not_called()
        self.assertEqual(hasher.calls, 0)

    async def test_capture_failure_returns_none_after(self) -> None:
        """
        Post-capture exceptions are absorbed and produce visual.after=None.
        """

        device = self.__device(swipe_result=ActionResult(success=True, duration=1))
        device.capture_screen = AsyncMock(side_effect=RuntimeError("capture exploded"))
        dispatcher = DeviceSwipeDispatcher(device=device, hasher=_FixedHasher(value="POST"))

        attempt = await dispatcher.attempt(path=self.__path(), index=1, original_before="PRE")

        self.assertTrue(attempt.device.succeeded)
        self.assertFalse(attempt.visual.changed)
        self.assertIsNone(attempt.visual.after)
