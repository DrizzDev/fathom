from __future__ import annotations

from logging import getLogger
from typing import Any, Dict, Optional

import httpx

from fathom.core.exceptions import DeviceError
from fathom.schemas.configuration import IOSConfiguration

logger = getLogger(__name__)


class IOSAutomationGateway:
    """
    WebDriverAgent client for iOS simulator hierarchy and gesture requests.
    """

    def __init__(self, *, configuration: IOSConfiguration) -> None:
        """
        Initialize iOS automation gateway configuration.
        """

        self.__configuration = configuration

    async def tap(self, *, x: float, y: float) -> None:
        """
        Tap a point through the configured iOS automation gateway.
        """

        session_identifier = await self.__create_session()

        try:
            await self.__request(
                method="POST",
                path=f"session/{session_identifier}/wda/tap",
                json_body={"x": round(x), "y": round(y)},
            )
        finally:
            await self.__delete_session(session_identifier=session_identifier)

    async def type_text(self, *, text: str) -> None:
        """
        Type text through the configured iOS automation gateway.
        """

        if not text:
            return

        session_identifier = await self.__create_session()

        try:
            await self.__request(
                method="POST",
                path=f"session/{session_identifier}/wda/keys",
                json_body={"value": list(text)},
            )
        finally:
            await self.__delete_session(session_identifier=session_identifier)

    async def swipe(
        self,
        *,
        start_x: float,
        start_y: float,
        end_x: float,
        end_y: float,
        duration_milliseconds: int,
    ) -> None:
        """
        Swipe between two points through the configured iOS automation gateway.
        """

        session_identifier = await self.__create_session()
        actions_path = f"session/{session_identifier}/actions"

        try:
            await self.__request(
                method="POST",
                path=actions_path,
                json_body={
                    "actions": [
                        {
                            "type": "pointer",
                            "id": "finger1",
                            "parameters": {"pointerType": "touch"},
                            "actions": [
                                {
                                    "type": "pointerMove",
                                    "duration": 0,
                                    "x": round(start_x),
                                    "y": round(start_y),
                                    "origin": "viewport",
                                },
                                {
                                    "type": "pointerDown",
                                    "button": 0,
                                },
                                {
                                    "type": "pointerMove",
                                    "duration": max(int(duration_milliseconds), 0),
                                    "x": round(end_x),
                                    "y": round(end_y),
                                    "origin": "viewport",
                                },
                                {
                                    "type": "pointerUp",
                                    "button": 0,
                                },
                            ],
                        }
                    ]
                },
            )
        finally:
            await self.__release_actions(actions_path=actions_path)
            await self.__delete_session(session_identifier=session_identifier)

    async def dump_source(self) -> str:
        """
        Fetch XML hierarchy source through the configured iOS automation gateway.
        """

        session_identifier = await self.__create_session()

        try:
            payload = await self.__request(
                method="GET",
                path=f"session/{session_identifier}/source?format=xml",
            )
            value = payload.get("value")

            if isinstance(value, str) and value.strip():
                return value

            raise DeviceError("WebDriverAgent returned an empty hierarchy source")
        finally:
            await self.__delete_session(session_identifier=session_identifier)

    async def get_window_size(self) -> tuple[int, int]:
        """
        Resolve iOS automation window size in logical points.
        """

        session_identifier = await self.__create_session()

        try:
            payload = await self.__request(
                method="GET",
                path=f"session/{session_identifier}/window/size",
            )
            value = payload.get("value")
            if not isinstance(value, dict):
                raise DeviceError("WebDriverAgent returned invalid window size payload")

            width = value.get("width")
            height = value.get("height")
            if not isinstance(width, int) or not isinstance(height, int):
                raise DeviceError("WebDriverAgent returned invalid window size fields")

            return width, height
        finally:
            await self.__delete_session(session_identifier=session_identifier)

    async def get_active_application_bundle_identifier(self) -> str | None:
        """
        Resolve the foreground bundle identifier through WebDriverAgent active app info.
        """

        payload = await self.__request(
            method="GET",
            path="wda/activeAppInfo",
        )
        value = payload.get("value")

        if not isinstance(value, dict):
            raise DeviceError("WebDriverAgent returned invalid active application payload")

        bundle_identifier = value.get("bundleId")
        if not isinstance(bundle_identifier, str) or not bundle_identifier.strip():
            return None

        return bundle_identifier.strip()

    async def press_home(self) -> None:
        """
        Trigger the iOS home screen through the automation gateway.
        """

        await self.__request(
            method="POST",
            path="wda/homescreen",
        )

    async def __create_session(self) -> str:
        """
        Create a short-lived automation session.
        """

        capabilities: Dict[str, Any] = {"platformName": "iOS"}
        bundle_identifier = (
            self.__configuration.web_driver_agent_bundle_identifier
            or self.__configuration.bundle_identifier
        )

        if bundle_identifier:
            capabilities["bundleId"] = bundle_identifier

        payload = await self.__request(
            method="POST",
            path="session",
            json_body={
                "capabilities": {
                    "alwaysMatch": capabilities,
                    "firstMatch": [{}],
                }
            },
        )

        session_identifier = payload.get("sessionId")
        if session_identifier is None:
            value = payload.get("value")
            if isinstance(value, dict):
                session_identifier = value.get("sessionId")

        if not isinstance(session_identifier, str) or not session_identifier:
            raise DeviceError("iOS automation session creation returned no sessionId")

        return session_identifier

    async def __delete_session(self, *, session_identifier: str) -> None:
        """
        Delete an automation session.
        """

        try:
            await self.__request(
                method="DELETE",
                path=f"session/{session_identifier}",
            )
        except Exception as exception:
            logger.warning(
                "Failed to delete iOS automation session %s: %s",
                session_identifier,
                exception,
            )

    async def __release_actions(self, *, actions_path: str) -> None:
        """
        Release active automation actions.
        """

        try:
            await self.__request(
                method="DELETE",
                path=actions_path,
            )
        except Exception as exception:
            logger.warning(
                "Failed to release iOS automation actions at %s: %s",
                actions_path,
                exception,
            )

    async def __request(
        self,
        *,
        method: str,
        path: str,
        json_body: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute an automation request and normalize the JSON payload.
        """

        base_url = self.__configuration.web_driver_agent_url.rstrip("/")
        timeout_seconds = self.__configuration.web_driver_agent_request_timeout_seconds
        url = f"{base_url}/{path.lstrip('/')}"

        try:
            async with httpx.AsyncClient(timeout=timeout_seconds) as client:
                response = await client.request(
                    method=method,
                    url=url,
                    json=json_body,
                )
                response.raise_for_status()
                payload = response.json()
        except httpx.HTTPStatusError as exception:
            raise DeviceError(
                f"iOS automation gateway {method} {path} failed with HTTP {exception.response.status_code}"
            ) from exception
        except httpx.HTTPError as exception:
            raise DeviceError(
                f"iOS automation gateway {method} {path} request failed: {exception}"
            ) from exception
        except Exception as exception:
            raise DeviceError(
                f"iOS automation gateway {method} {path} failed: {exception}"
            ) from exception

        if isinstance(payload, dict):
            return payload

        return {"value": payload}
