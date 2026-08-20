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


class ADBDeviceTouchInputTest(unittest.IsolatedAsyncioTestCase):
    """
    Cover local ADB touch source routing for pointer gestures.
    """

    def __build_device(self) -> ADBDevice:
        """
        Build a local ADB device with default configuration.
        """

        return ADBDevice(serial="emulator-5554")

    async def test_tap_uses_explicit_touchscreen_source(self) -> None:
        """
        Taps must be routed as touchscreen input even when an edit field owns focus.
        """

        device = self.__build_device()

        with patch.object(
            device,
            "_ADBDevice__shell",
            new_callable=AsyncMock,
            return_value=ActionResult(success=True, duration=1),
        ) as mock_shell:
            result = await device.tap(x=123, y=456)

        self.assertTrue(result.success)
        mock_shell.assert_awaited_once_with(command="input touchscreen tap 123 456")

    async def test_swipe_uses_explicit_touchscreen_source(self) -> None:
        """
        Swipes must be routed as touchscreen input instead of falling back to input defaults.
        """

        device = self.__build_device()

        with patch.object(
            device,
            "_ADBDevice__shell",
            new_callable=AsyncMock,
            return_value=ActionResult(success=True, duration=1),
        ) as mock_shell:
            result = await device.swipe(x1=1, y1=2, x2=3, y2=4, duration=500)

        self.assertTrue(result.success)
        mock_shell.assert_awaited_once_with(command="input touchscreen swipe 1 2 3 4 500")


class ADBDeviceHierarchyDumpTest(unittest.IsolatedAsyncioTestCase):
    """
    Cover Android UiAutomation recovery around hierarchy dumps.
    """

    def __build_device(self) -> ADBDevice:
        """
        Build a local ADB device with default configuration.
        """

        return ADBDevice(serial="emulator-5554")

    async def test_dump_hierarchy_recovers_stale_instrumentation_before_dump(self) -> None:
        """
        Active UiAutomation state must kill stale shell instrumentation
        before invoking ``uiautomator dump``.
        """

        device = self.__build_device()
        ok = ActionResult(success=True, duration=1)
        shell_results = [
            ActionResult(success=True, duration=1, output="Ui Automation[eventTypes=TYPES_ALL]"),
            ok,
            ok,
            ok,
        ]

        with (
            patch.object(
                device, "_ADBDevice__shell", new_callable=AsyncMock, side_effect=shell_results
            ) as mock_shell,
            patch.object(
                device,
                "_ADBDevice__run_safe_subprocess",
                new_callable=AsyncMock,
                return_value=(0, b"<hierarchy />", b""),
            ),
        ):
            result = await device.dump_hierarchy()

        self.assertEqual(result, "<hierarchy />")
        commands = [
            c.kwargs.get("command", c.args[0] if c.args else "") for c in mock_shell.call_args_list
        ]
        self.assertEqual(commands[0], "dumpsys accessibility")
        self.assertIn("pidof app_process", commands[1])
        self.assertIn("com.android.commands.am.Am instrument", commands[1])
        self.assertIn("kill -9", commands[1])
        self.assertIn("rm -f /data/local/tmp/window_dump.xml", commands[2])
        self.assertIn("uiautomator dump --compressed", commands[3])

    async def test_dump_hierarchy_recovers_after_compressed_dump_failure(self) -> None:
        """
        Failed compressed dump must recover stale UiAutomation before the
        uncompressed fallback; it must not call ``pkill uiautomator``.
        """

        device = self.__build_device()
        ok = ActionResult(success=True, duration=1)
        shell_results = [
            ActionResult(success=True, duration=1, output=""),
            ok,
            ActionResult(success=False, duration=1, error="ADB shell command exited with code 137"),
            ActionResult(success=True, duration=1, output="Ui Automation[eventTypes=TYPES_ALL]"),
            ok,
            ok,
        ]

        with (
            patch.object(
                device, "_ADBDevice__shell", new_callable=AsyncMock, side_effect=shell_results
            ) as mock_shell,
            patch.object(
                device,
                "_ADBDevice__run_safe_subprocess",
                new_callable=AsyncMock,
                return_value=(0, b"<hierarchy />", b""),
            ),
        ):
            result = await device.dump_hierarchy()

        self.assertEqual(result, "<hierarchy />")
        commands = [
            c.kwargs.get("command", c.args[0] if c.args else "") for c in mock_shell.call_args_list
        ]
        self.assertIn("uiautomator dump --compressed", commands[2])
        self.assertEqual(commands[3], "dumpsys accessibility")
        self.assertIn("pidof app_process", commands[4])
        self.assertIn("uiautomator dump /data/local/tmp/window_dump.xml", commands[5])
        self.assertFalse(any("pkill -9 uiautomator" in command for command in commands))

    async def test_dump_hierarchy_does_not_fallback_after_timeout(self) -> None:
        """
        A timed-out compressed dump must fail fast instead of entering the fallback ladder.
        """

        device = self.__build_device()
        shell_results = [
            ActionResult(success=True, duration=1, output=""),
            ActionResult(success=True, duration=1),
            ActionResult(
                success=False,
                duration=0,
                error=(
                    "Command timed out after 10.0s: adb -s emulator-5554 shell "
                    "uiautomator dump --compressed /data/local/tmp/window_dump.xml"
                ),
            ),
        ]

        with (
            patch.object(
                device, "_ADBDevice__shell", new_callable=AsyncMock, side_effect=shell_results
            ) as mock_shell,
            self.assertRaisesRegex(Exception, "compressed UI automation dump timed out"),
        ):
            await device.dump_hierarchy()

        commands = [
            c.kwargs.get("command", c.args[0] if c.args else "") for c in mock_shell.call_args_list
        ]
        self.assertEqual(len(commands), 3)
        self.assertIn("uiautomator dump --compressed", commands[2])
        self.assertFalse(
            any(
                command == "uiautomator dump /data/local/tmp/window_dump.xml"
                for command in commands
            )
        )


class ADBDeviceFrameParseTest(unittest.TestCase):
    """
    Verify focused-window frame parsing from a real window-manager dump shape.
    """

    __DUMP = """
WINDOW MANAGER WINDOWS (dumpsys window windows)
  Window #0 Window{1a2b3c4 u0 NavigationBar0}:
    mFrame=[0,2148][1080,2208]
  Window #1 Window{5d6e7f8 u0 com.delivery.android/com.delivery.android.MainActivity}:
    mDisplayId=0 rootTaskId=1
    mFrame=[0,80][1080,2080]
  Window #2 Window{9a8b7c6 u0 StatusBar}:
    mFrame=[0,0][1080,80]
  mCurrentFocus=Window{5d6e7f8 u0 com.delivery.android/com.delivery.android.MainActivity}
"""

    def test_parses_focused_window_frame(self) -> None:
        """
        The focused app window's frame is returned, not the bars'.
        """

        frame = ADBDevice._frame_from(dump=self.__DUMP)

        self.assertIsNotNone(frame)
        assert frame is not None
        self.assertEqual((frame.x, frame.y, frame.width, frame.height), (0, 80, 1080, 2000))

    def test_unparseable_dump_fails_open_to_none(self) -> None:
        """
        Any dump without a matching focused window fails open to None.
        """

        self.assertIsNone(ADBDevice._frame_from(dump="mCurrentFocus=null"))
        self.assertIsNone(ADBDevice._frame_from(dump=""))
