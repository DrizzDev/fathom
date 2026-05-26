"""
Unit pins for :class:`RemoteDeviceConfiguration.request_timeout` wiring.

Remote runs route every device interaction (screenshot, hierarchy dump,
action dispatch) through HTTP, so the per-request budget is the only
control surface for hangs there. These tests verify that the configured
value reaches :class:`httpx.AsyncClient` at construction time and that
the documented default is preserved.
"""

from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from fathom.adapters.device.remote.adb import ADBRemoteDeviceAdapter
from fathom.constants.platform import DeviceConnectionType, DevicePlatform
from fathom.schemas.configuration import DeviceConfiguration, RemoteDeviceConfiguration


class RemoteRequestTimeoutWiringTest(unittest.TestCase):
    """
    Pin that ``RemoteDeviceConfiguration.request_timeout`` flows into the HTTP client.
    """

    @staticmethod
    def __build_configuration(*, request_timeout: float) -> DeviceConfiguration:
        """
        Build a remote Android configuration with the requested timeout.
        """

        return DeviceConfiguration(
            type=DeviceConnectionType.REMOTE,
            platform=DevicePlatform.ANDROID,
            remote=RemoteDeviceConfiguration(
                session_id="session-id",
                provider_url="https://example.test",
                request_timeout=request_timeout,
            ),
        )

    def test_default_timeout_is_sixty_seconds(self) -> None:
        """
        Documented default keeps long-tail snapshot transfers safe.
        """

        configuration = RemoteDeviceConfiguration()
        self.assertEqual(configuration.request_timeout, 60.0)

    def test_custom_timeout_passes_to_httpx_async_client(self) -> None:
        """
        A configured ``request_timeout`` is the value handed to httpx.
        """

        client = Mock()
        client.is_closed = False

        configuration = self.__build_configuration(request_timeout=42.5)

        with patch(
            "fathom.adapters.device.remote.adb.httpx.AsyncClient",
            return_value=client,
        ) as construct:
            ADBRemoteDeviceAdapter(configuration=configuration)

        self.assertEqual(construct.call_args.kwargs["timeout"], 42.5)

    def test_default_remote_configuration_uses_sixty_second_client_timeout(self) -> None:
        """
        Without overrides, the HTTP client gets the 60-second default.
        """

        client = Mock()
        client.is_closed = False

        configuration = DeviceConfiguration(
            type=DeviceConnectionType.REMOTE,
            platform=DevicePlatform.ANDROID,
            remote=RemoteDeviceConfiguration(
                session_id="session-id",
                provider_url="https://example.test",
            ),
        )

        with patch(
            "fathom.adapters.device.remote.adb.httpx.AsyncClient",
            return_value=client,
        ) as construct:
            ADBRemoteDeviceAdapter(configuration=configuration)

        self.assertEqual(construct.call_args.kwargs["timeout"], 60.0)
