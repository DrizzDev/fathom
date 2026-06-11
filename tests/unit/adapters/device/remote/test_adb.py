from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, Mock, patch

from fathom.adapters.device.remote.adb import ADBRemoteDeviceAdapter
from fathom.constants.platform import DeviceConnectionType, DevicePlatform
from fathom.core.exceptions import DeviceConnectionClosedError
from fathom.schemas.configuration import DeviceConfiguration, RemoteDeviceConfiguration


class ADBRemoteDeviceAdapterTest(unittest.IsolatedAsyncioTestCase):
    """
    Cover error handling for the remote Android transport adapter.
    """

    def __build_configuration(self) -> DeviceConfiguration:
        """
        Create a minimal remote Android configuration for tests.
        """

        return DeviceConfiguration(
            type=DeviceConnectionType.REMOTE,
            platform=DevicePlatform.ANDROID,
            remote=RemoteDeviceConfiguration(
                session_id="session-id",
                provider_url="https://example.test",
            ),
        )

    async def test_get_snapshot_fails_fast_when_client_is_already_closed(self) -> None:
        """
        Do not retry snapshot requests once the shared client has been closed.
        """

        client = Mock()
        client.is_closed = True
        client.request = AsyncMock()

        with patch("fathom.adapters.device.remote.adb.httpx.AsyncClient", return_value=client):
            adapter = ADBRemoteDeviceAdapter(configuration=self.__build_configuration())

        with self.assertRaises(DeviceConnectionClosedError) as context:
            await adapter.get_snapshot()

        self.assertEqual(
            context.exception.display(
                fallback="Failed to capture the current app screen. Please retry.",
            ),
            "Lost the device connection. Please retry the run.",
        )
        self.assertFalse(context.exception.retryable)
        client.request.assert_not_awaited()

    async def test_get_snapshot_marks_closed_client_runtime_errors_as_non_retryable(self) -> None:
        """
        Collapse late closed-client RuntimeErrors into a single non-retryable DeviceError.
        """

        client = Mock()
        client.is_closed = False
        client.request = AsyncMock(
            side_effect=RuntimeError("Cannot send a request, as the client has been closed.")
        )

        with patch("fathom.adapters.device.remote.adb.httpx.AsyncClient", return_value=client):
            adapter = ADBRemoteDeviceAdapter(configuration=self.__build_configuration())

        with self.assertRaises(DeviceConnectionClosedError) as context:
            await adapter.get_snapshot()

        self.assertEqual(
            context.exception.display(
                fallback="Failed to capture the current app screen. Please retry.",
            ),
            "Lost the device connection. Please retry the run.",
        )
        self.assertFalse(context.exception.retryable)
        self.assertEqual(client.request.await_count, 1)

    async def test_get_snapshot_uses_shared_remote_timeout(self) -> None:
        """
        Use the shared remote-client timeout instead of per-request overrides.
        """

        response = Mock()
        response.content = (4).to_bytes(4, byteorder="big") + b"abcd"
        response.raise_for_status = Mock()

        client = Mock()
        client.is_closed = False
        client.request = AsyncMock(return_value=response)

        with patch(
            "fathom.adapters.device.remote.adb.httpx.AsyncClient",
            return_value=client,
        ) as client_factory:
            adapter = ADBRemoteDeviceAdapter(configuration=self.__build_configuration())

        image_bytes, xml_content = await adapter.get_snapshot()

        self.assertEqual(image_bytes, b"abcd")
        self.assertEqual(xml_content, "")
        client_factory.assert_called_once()
        self.assertEqual(client_factory.call_args.kwargs["timeout"], 60.0)
        client.request.assert_awaited_once_with("POST", "snapshot", params={})
