from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

from fathom.adapters.device.local.ios import IOSDevice


class IOSDeviceLaunchPackageTest(unittest.IsolatedAsyncioTestCase):
    """
    Cover launching an application by bundle identifier via simctl.
    """

    async def test_launch_invokes_simctl_launch_with_bundle(self) -> None:
        """
        Launching a bundle resolves the device and runs simctl launch for it.
        """

        device = IOSDevice()
        with (
            patch.object(
                device,
                "_IOSDevice__resolve_device_identifier",
                new_callable=AsyncMock,
                return_value="UDID-1",
            ),
            patch.object(
                device,
                "_IOSDevice__run_simctl",
                new_callable=AsyncMock,
                return_value=(0, b"", b""),
            ) as mock_simctl,
        ):
            result = await device.launch_package(package_name="com.apple.Preferences")

        self.assertTrue(result.success)
        self.assertEqual(
            mock_simctl.await_args.kwargs["parts"],
            ["launch", "UDID-1", "com.apple.Preferences"],
        )
