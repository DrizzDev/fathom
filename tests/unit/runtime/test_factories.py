from __future__ import annotations

import unittest
from unittest.mock import patch

from fathom.adapters.device.remote.adb import ADBRemoteDeviceAdapter
from fathom.adapters.device.remote.ios import IOSRemoteDeviceAdapter
from fathom.constants.platform import DeviceConnectionType, DevicePlatform
from fathom.runtime.factories import DeviceFactory
from fathom.schemas.configuration import DeviceConfiguration, RemoteDeviceConfiguration


class DeviceFactoryTest(unittest.TestCase):
    """
    Keep Android and iOS remote device selection distinct.
    """

    def test_remote_ios_uses_ios_remote_adapter(self) -> None:
        """
        Select the iOS-specific remote adapter for remote iOS runs.
        """

        factory = DeviceFactory()
        with patch("fathom.adapters.device.remote.adb.httpx.AsyncClient"):
            device = factory.create(
                configuration=DeviceConfiguration(
                    type=DeviceConnectionType.REMOTE,
                    platform=DevicePlatform.IOS,
                    remote=RemoteDeviceConfiguration(
                        session_id="session-id",
                        provider_url="https://example.test",
                    ),
                )
            )

        self.assertIsInstance(device, IOSRemoteDeviceAdapter)

    def test_remote_android_uses_android_remote_adapter(self) -> None:
        """
        Keep Android remote runs on the transport-only adapter.
        """

        factory = DeviceFactory()
        with patch("fathom.adapters.device.remote.adb.httpx.AsyncClient"):
            device = factory.create(
                configuration=DeviceConfiguration(
                    type=DeviceConnectionType.REMOTE,
                    platform=DevicePlatform.ANDROID,
                    remote=RemoteDeviceConfiguration(
                        provider_url="https://example.test",
                        session_id="session-id",
                    ),
                )
            )

        self.assertIsInstance(device, ADBRemoteDeviceAdapter)
