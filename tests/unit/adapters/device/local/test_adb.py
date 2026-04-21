from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

from fathom.adapters.device.local.adb import ADBDevice
from fathom.constants.platform import AndroidClearStrategy, AndroidKeycode
from fathom.schemas.results import ActionResult


class ADBDeviceClearTest(unittest.IsolatedAsyncioTestCase):
    """
    Cover the local ADB text-clear strategies: modern Ctrl+A on SDK 30+
    and legacy batched-delete fallback for older devices.
    """

    def __build_device(self) -> ADBDevice:
        """
        Build a local ADB device with default configuration.
        """

        return ADBDevice(serial="emulator-5554")

    async def test_modern_clear_uses_ctrl_a_select_then_delete(self) -> None:
        """
        On SDK 30+, clear sends keycombination Ctrl+A then two DEL keyevents.
        """

        device = self.__build_device()
        ok = ActionResult(success=True, duration=1)
        shell_results = [
            ActionResult(success=True, duration=1, output="31"),  # getprop sdk
            ok,  # keycombination Ctrl+A
            ok,  # keyevent DEL DEL
            ok,  # type char 'x'
        ]

        with patch.object(
            device, "_ADBDevice__shell", new_callable=AsyncMock, side_effect=shell_results
        ) as mock_shell:
            result = await device.type(text="x", replace=True, prefilled="old text")

        self.assertTrue(result.success)
        calls = [
            c.kwargs.get("command", c.args[0] if c.args else "") for c in mock_shell.call_args_list
        ]

        self.assertIn("getprop ro.build.version.sdk", calls[0])
        self.assertIn(
            f"input keycombination {AndroidKeycode.CTRL_LEFT} {AndroidKeycode.A}", calls[1]
        )
        self.assertIn(f"input keyevent {AndroidKeycode.DEL} {AndroidKeycode.DEL}", calls[2])

    async def test_legacy_clear_uses_move_end_and_batched_deletes(self) -> None:
        """
        On SDK < 30, clear moves cursor to end then sends batched DEL keyevents.
        """

        device = self.__build_device()
        ok = ActionResult(success=True, duration=1)
        shell_results = [
            ActionResult(success=True, duration=1, output="28"),  # getprop sdk
            ok,  # move end + right arrows
            ok,  # batched deletes
            ok,  # type char 'x'
        ]

        with patch.object(
            device, "_ADBDevice__shell", new_callable=AsyncMock, side_effect=shell_results
        ) as mock_shell:
            result = await device.type(text="x", replace=True, prefilled="old text")

        self.assertTrue(result.success)
        calls = [
            c.kwargs.get("command", c.args[0] if c.args else "") for c in mock_shell.call_args_list
        ]

        # Legacy path: move cursor to end + right arrows
        self.assertIn(str(AndroidKeycode.MOVE_END), calls[1])
        self.assertEqual(
            calls[1].count(str(AndroidKeycode.DPAD_RIGHT)),
            AndroidClearStrategy.RIGHT_ARROW_COUNT,
        )
        # Batched deletes
        self.assertEqual(
            calls[2].count(str(AndroidKeycode.DEL)),
            AndroidClearStrategy.DELETE_COUNT,
        )

    async def test_modern_clear_falls_back_to_legacy_on_failure(self) -> None:
        """
        When Ctrl+A fails on SDK 30+, falls back to the legacy strategy.
        """

        device = self.__build_device()
        ok = ActionResult(success=True, duration=1)
        shell_results = [
            ActionResult(success=True, duration=1, output="33"),  # getprop sdk
            ActionResult(success=False, error="keycombination failed", duration=1),  # Ctrl+A fails
            ok,  # legacy move end
            ok,  # legacy deletes
            ok,  # type char 'x'
        ]

        with patch.object(
            device, "_ADBDevice__shell", new_callable=AsyncMock, side_effect=shell_results
        ) as mock_shell:
            result = await device.type(text="x", replace=True, prefilled="old")

        self.assertTrue(result.success)
        self.assertEqual(mock_shell.await_count, 5)

    async def test_no_clear_when_prefilled_is_empty(self) -> None:
        """
        When prefilled is empty, no clear is attempted.
        """

        device = self.__build_device()

        with patch.object(
            device,
            "_ADBDevice__shell",
            new_callable=AsyncMock,
            return_value=ActionResult(success=True, duration=1),
        ) as mock_shell:
            await device.type(text="a", replace=True, prefilled="")

        # Only type calls, no getprop or clear
        commands = [
            c.kwargs.get("command", c.args[0] if c.args else "") for c in mock_shell.call_args_list
        ]
        self.assertTrue(all("getprop" not in cmd for cmd in commands))
