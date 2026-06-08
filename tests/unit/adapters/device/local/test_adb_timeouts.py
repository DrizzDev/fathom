from __future__ import annotations

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from fathom.adapters.device.local.adb import ADBDevice
from fathom.core.exceptions import DeviceError
from fathom.schemas.configuration import ADBConfiguration


class ADBSnapshotTimeoutTest(unittest.IsolatedAsyncioTestCase):
    """
    Pins ``get_snapshot`` behavior when optional hierarchy capture is slow.
    """

    def __build_device(self, *, snapshot_timeout: float) -> ADBDevice:
        """
        Build a device whose snapshot timeout is set explicitly low for the test.
        """

        return ADBDevice(
            configuration=ADBConfiguration(
                serial_number="emulator-5554",
                snapshot_timeout=snapshot_timeout,
            ),
        )

    async def test_snapshot_returns_image_when_hierarchy_times_out(self) -> None:
        """
        A slow hierarchy dump is optional and should not discard the screenshot.
        """

        device = self.__build_device(snapshot_timeout=0.1)

        async def __forever_str() -> str:
            await asyncio.sleep(10)
            return ""

        with (
            patch.object(device, "capture_screen", new=AsyncMock(return_value=b"png-bytes")),
            patch.object(device, "dump_hierarchy", new=AsyncMock(side_effect=__forever_str)),
        ):
            image, xml = await device.get_snapshot()

        self.assertIsNone(xml)
        self.assertEqual(image, b"png-bytes")

    async def test_snapshot_returns_when_both_paths_succeed(self) -> None:
        """
        Happy path: when both screencap and hierarchy dump return inside the
        budget, ``get_snapshot`` returns the merged tuple.
        """

        device = self.__build_device(snapshot_timeout=1.0)

        with (
            patch.object(device, "capture_screen", new=AsyncMock(return_value=b"png-bytes")),
            patch.object(device, "dump_hierarchy", new=AsyncMock(return_value="<xml/>")),
        ):
            image, xml = await device.get_snapshot()

        self.assertEqual(xml, "<xml/>")
        self.assertEqual(image, b"png-bytes")


class ADBHierarchyLockTimeoutTest(unittest.IsolatedAsyncioTestCase):
    """
    Pins the ``hierarchy_lock_timeout`` guard that protects against a leaked
    lock from a prior cancelled ``dump_hierarchy`` task.
    """

    def __build_device(self, *, hierarchy_lock_timeout: float) -> ADBDevice:
        return ADBDevice(
            configuration=ADBConfiguration(
                serial_number="emulator-5554",
                hierarchy_lock_timeout=hierarchy_lock_timeout,
            ),
        )

    async def test_lock_timeout_raises_when_lock_is_held(self) -> None:
        """
        When the lock is held by a forgotten holder, the next acquire fails fast.
        """

        device = self.__build_device(hierarchy_lock_timeout=0.1)

        # Simulate a leaked lock by acquiring it from a parallel task
        # that never releases it within this test's window.
        leaked = asyncio.Event()
        release_signal = asyncio.Event()

        async def __hold_lock() -> None:
            async with device._ADBDevice__hierarchy_lock:  # type: ignore[attr-defined]
                leaked.set()
                await release_signal.wait()

        holder = asyncio.create_task(__hold_lock())
        await leaked.wait()

        try:
            with self.assertRaises(DeviceError) as context:
                await device.dump_hierarchy()

            self.assertIn("lock acquire timed out", str(context.exception))

        finally:
            release_signal.set()
            await holder

    async def test_lock_is_released_after_normal_dump(self) -> None:
        """
        On a successful dump, the lock is released so the next call can acquire.
        """

        device = self.__build_device(hierarchy_lock_timeout=1.0)

        with patch.object(
            device,
            "_ADBDevice__dump_hierarchy_locked",
            new=AsyncMock(return_value="<xml/>"),
        ):
            first = await device.dump_hierarchy()
            second = await device.dump_hierarchy()

        self.assertEqual(first, "<xml/>")
        self.assertEqual(second, "<xml/>")
        self.assertFalse(device._ADBDevice__hierarchy_lock.locked())  # type: ignore[attr-defined]

    async def test_lock_is_released_when_locked_dump_raises(self) -> None:
        """
        If the dump itself fails, the lock must still be released.
        """

        device = self.__build_device(hierarchy_lock_timeout=1.0)

        with (
            patch.object(
                device,
                "_ADBDevice__dump_hierarchy_locked",
                new=AsyncMock(side_effect=DeviceError("dump failed")),
            ),
            self.assertRaises(DeviceError),
        ):
            await device.dump_hierarchy()

        self.assertFalse(device._ADBDevice__hierarchy_lock.locked())  # type: ignore[attr-defined]


class ADBSubprocessCleanupTimeoutTest(unittest.IsolatedAsyncioTestCase):
    """
    Pins the bounded post-kill subprocess reap. A subprocess stuck in
    uninterruptible IO will not exit after SIGKILL until the kernel can
    deliver the signal; awaiting its ``wait()`` would block the loop
    indefinitely. The cleanup helper caps this with
    ``subprocess_cleanup_timeout`` and abandons the process.
    """

    def __build_device(self, *, subprocess_cleanup_timeout: float) -> ADBDevice:
        return ADBDevice(
            configuration=ADBConfiguration(
                serial_number="emulator-5554",
                subprocess_cleanup_timeout=subprocess_cleanup_timeout,
            ),
        )

    async def test_cleanup_abandons_unkillable_process(self) -> None:
        """
        When the process never reaps after kill, the helper logs and returns
        rather than awaiting indefinitely.
        """

        device = self.__build_device(subprocess_cleanup_timeout=0.1)

        process = MagicMock(spec=asyncio.subprocess.Process)
        process.pid = 99999
        process.kill = MagicMock()

        async def __never_reaps() -> int:
            await asyncio.sleep(10)
            return 0

        process.wait = AsyncMock(side_effect=__never_reaps)

        await device._ADBDevice__abandon_unkillable_subprocess(  # type: ignore[attr-defined]
            process=process, arguments=["adb", "shell", "uiautomator", "dump"]
        )

        process.kill.assert_called_once()
        process.wait.assert_awaited()

    async def test_cleanup_reaps_when_process_exits(self) -> None:
        """
        Happy path: when the process exits inside the budget, the helper
        completes without warning.
        """

        device = self.__build_device(subprocess_cleanup_timeout=1.0)

        process = MagicMock(spec=asyncio.subprocess.Process)
        process.pid = 99998
        process.kill = MagicMock()
        process.wait = AsyncMock(return_value=-9)

        await device._ADBDevice__abandon_unkillable_subprocess(  # type: ignore[attr-defined]
            process=process, arguments=["adb", "shell", "ls"]
        )

        process.kill.assert_called_once()
        process.wait.assert_awaited()

    async def test_cleanup_swallows_already_dead_process(self) -> None:
        """
        :class:`ProcessLookupError` on kill is benign and must not raise.
        """

        device = self.__build_device(subprocess_cleanup_timeout=1.0)

        process = MagicMock(spec=asyncio.subprocess.Process)
        process.pid = 99997
        process.kill = MagicMock(side_effect=ProcessLookupError)
        process.wait = AsyncMock()

        # Must not raise.
        await device._ADBDevice__abandon_unkillable_subprocess(  # type: ignore[attr-defined]
            process=process, arguments=["adb", "shell", "ls"]
        )

        process.wait.assert_not_awaited()
