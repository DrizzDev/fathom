from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

from fathom.adapters.ios.gateway import IOSAutomationGateway
from fathom.constants.platform import IOSClearStrategy
from fathom.schemas.configuration import IOSConfiguration


class IOSAutomationGatewayClearTest(unittest.IsolatedAsyncioTestCase):
    """
    Cover the WDA-backed text clearing through the iOS automation gateway.
    """

    @staticmethod
    def __build_gateway() -> IOSAutomationGateway:
        """
        Build a gateway with default WDA configuration.
        """

        return IOSAutomationGateway(
            configuration=IOSConfiguration(
                bundle_id="com.test.app",
                wda_url="http://localhost:8100",
            )
        )

    async def test_clear_posts_backspace_array_to_wda_keys(self) -> None:
        """
        clear_text sends a POST to /session/{id}/wda/keys with backspace characters.
        """

        gateway = self.__build_gateway()

        with (
            patch.object(
                gateway, "_IOSAutomationGateway__create_session", new_callable=AsyncMock, return_value="sess-1"
            ),
            patch.object(
                gateway, "_IOSAutomationGateway__delete_session", new_callable=AsyncMock
            ),
            patch.object(
                gateway, "_IOSAutomationGateway__request", new_callable=AsyncMock
            ) as mock_request,
        ):
            await gateway.clear_text(length=5)

        mock_request.assert_awaited_once()
        call_kwargs = mock_request.call_args.kwargs

        self.assertEqual(call_kwargs["method"], "POST")
        self.assertIn("wda/keys", call_kwargs["path"])
        self.assertEqual(call_kwargs["json_body"]["value"], ["\b"] * 5)

    async def test_clear_clamps_length_to_max(self) -> None:
        """
        Lengths exceeding the safety clamp are capped.
        """

        gateway = self.__build_gateway()

        with (
            patch.object(
                gateway, "_IOSAutomationGateway__create_session", new_callable=AsyncMock, return_value="sess-1"
            ),
            patch.object(
                gateway, "_IOSAutomationGateway__delete_session", new_callable=AsyncMock
            ),
            patch.object(
                gateway, "_IOSAutomationGateway__request", new_callable=AsyncMock
            ) as mock_request,
        ):
            await gateway.clear_text(length=99_999)

        payload = mock_request.call_args.kwargs["json_body"]["value"]
        self.assertEqual(len(payload), IOSClearStrategy.MAX_LENGTH)

    async def test_clear_skips_for_zero_length(self) -> None:
        """
        No WDA call is made when length is zero or negative.
        """

        gateway = self.__build_gateway()

        with (
            patch.object(
                gateway, "_IOSAutomationGateway__create_session", new_callable=AsyncMock
            ) as mock_session,
            patch.object(
                gateway, "_IOSAutomationGateway__request", new_callable=AsyncMock
            ) as mock_request,
        ):
            await gateway.clear_text(length=0)
            await gateway.clear_text(length=-5)

        mock_session.assert_not_awaited()
        mock_request.assert_not_awaited()
