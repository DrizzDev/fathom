from __future__ import annotations

import unittest
from typing import List
from unittest.mock import AsyncMock

from fathom.adapters.device.local.adb import ADBDevice
from fathom.schemas.configuration import ADBConfiguration


class ADBDeviceLaunchConfiguredPackageTest(unittest.IsolatedAsyncioTestCase):
    """
    Cover ``ADBDevice.launch_configured_package`` — the one-shot
    auto-launch hook the intent strategy fires alongside the LLM
    classifier so the app is ready by the first capture.
    """

    async def test_invokes_monkey_with_package_and_launcher_category(self) -> None:
        device = ADBDevice(
            configuration=ADBConfiguration(
                serial_number="EMULATOR-5554",
                package_name="com.example.app",
            )
        )

        captured_argv: List[List[str]] = []

        async def fake_subprocess(*, arguments=None, **_kwargs):  # type: ignore[no-untyped-def]
            captured_argv.append(list(arguments))
            return 0, b"events: 1", b""

        device._ADBDevice__run_safe_subprocess = fake_subprocess  # type: ignore[attr-defined]

        await device.launch_configured_package()

        self.assertEqual(len(captured_argv), 1)
        argv = captured_argv[0]
        self.assertIn("monkey", argv)
        self.assertIn("-p", argv)
        self.assertIn("com.example.app", argv)
        self.assertIn("android.intent.category.LAUNCHER", argv)
        self.assertIn("1", argv)
        # Serial must be forwarded via `-s`.
        self.assertIn("-s", argv)
        self.assertIn("EMULATOR-5554", argv)

    async def test_no_op_when_package_not_configured(self) -> None:
        device = ADBDevice(configuration=ADBConfiguration(serial_number="EMULATOR-5554"))
        device._ADBDevice__run_safe_subprocess = AsyncMock()  # type: ignore[attr-defined]

        await device.launch_configured_package()

        device._ADBDevice__run_safe_subprocess.assert_not_awaited()  # type: ignore[attr-defined]

    async def test_idempotent_on_repeated_calls(self) -> None:
        device = ADBDevice(
            configuration=ADBConfiguration(
                serial_number="EMULATOR-5554",
                package_name="com.example.app",
            )
        )
        call_count = 0

        async def fake_subprocess(**_kwargs):  # type: ignore[no-untyped-def]
            nonlocal call_count
            call_count += 1
            return 0, b"events: 1", b""

        device._ADBDevice__run_safe_subprocess = fake_subprocess  # type: ignore[attr-defined]

        await device.launch_configured_package()
        await device.launch_configured_package()
        await device.launch_configured_package()

        # Guarded: monkey fires once even if the launch method is
        # called multiple times.
        self.assertEqual(call_count, 1)

    async def test_swallows_subprocess_failure(self) -> None:
        """A launch failure (e.g., package not installed) must not
        propagate — the agent should still be able to navigate from
        the launcher screen."""

        device = ADBDevice(
            configuration=ADBConfiguration(
                serial_number="EMULATOR-5554",
                package_name="com.missing.app",
            )
        )

        async def fake_subprocess(**_kwargs):  # type: ignore[no-untyped-def]
            return 1, b"", b"Error: package not found"

        device._ADBDevice__run_safe_subprocess = fake_subprocess  # type: ignore[attr-defined]

        # Must not raise.
        await device.launch_configured_package()


class ADBDeviceTerminateConfiguredPackageTest(unittest.IsolatedAsyncioTestCase):
    """
    Cover the symmetric teardown hook fired from the intent strategy's
    ``finally`` block so the picked app is force-stopped on run exit.
    """

    async def test_invokes_am_force_stop_with_package(self) -> None:
        device = ADBDevice(
            configuration=ADBConfiguration(
                serial_number="EMULATOR-5554",
                package_name="com.example.app",
            )
        )

        captured_argv: List[List[str]] = []

        async def fake_subprocess(*, arguments=None, **_kwargs):  # type: ignore[no-untyped-def]
            captured_argv.append(list(arguments))
            return 0, b"", b""

        device._ADBDevice__run_safe_subprocess = fake_subprocess  # type: ignore[attr-defined]

        await device.terminate_configured_package()

        self.assertEqual(len(captured_argv), 1)
        argv = captured_argv[0]
        self.assertIn("am", argv)
        self.assertIn("force-stop", argv)
        self.assertIn("com.example.app", argv)
        self.assertIn("-s", argv)
        self.assertIn("EMULATOR-5554", argv)

    async def test_no_op_when_package_not_configured(self) -> None:
        device = ADBDevice(configuration=ADBConfiguration(serial_number="EMULATOR-5554"))
        device._ADBDevice__run_safe_subprocess = AsyncMock()  # type: ignore[attr-defined]

        await device.terminate_configured_package()

        device._ADBDevice__run_safe_subprocess.assert_not_awaited()  # type: ignore[attr-defined]

    async def test_swallows_subprocess_failure(self) -> None:
        device = ADBDevice(
            configuration=ADBConfiguration(
                serial_number="EMULATOR-5554",
                package_name="com.missing.app",
            )
        )

        async def fake_subprocess(**_kwargs):  # type: ignore[no-untyped-def]
            return 1, b"", b"Error: not running"

        device._ADBDevice__run_safe_subprocess = fake_subprocess  # type: ignore[attr-defined]

        # Must not raise even when the subprocess reports a non-zero exit.
        await device.terminate_configured_package()


class ADBDeviceInterfaceDefaultTest(unittest.IsolatedAsyncioTestCase):
    """
    Verify the ``DevicePort.launch_configured_package`` default (no-op)
    is available on ``ADBDevice`` via the interface contract.
    """

    async def test_default_method_on_device_port_is_async_and_no_op(self) -> None:
        from fathom.interfaces.device import DevicePort

        # DevicePort.launch_configured_package is a concrete default
        # on the base class; calling it via an ADBDevice without a
        # configured package exercises the same path as any adapter
        # that didn't override it.
        device = ADBDevice(configuration=ADBConfiguration(serial_number="EMU"))
        result = await DevicePort.launch_configured_package(device)  # type: ignore[arg-type]
        self.assertIsNone(result)
