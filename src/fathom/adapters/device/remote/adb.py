from __future__ import annotations

import asyncio
import base64
import logging
import struct
import time
from logging import getLogger
from typing import Any, Dict, Optional, Tuple, cast
from urllib.parse import urljoin

import httpx
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from fathom.constants.execution import REMOTE_DEVICE_REQUEST_TIMEOUT_SECONDS
from fathom.constants.interaction import InteractionAction, SwipeSpeed
from fathom.constants.platform import DevicePlatform
from fathom.core.exceptions import (
    DeviceConnectionClosedError,
    DeviceError,
    FathomError,
    PortError,
)
from fathom.interfaces.device import DevicePort
from fathom.schemas.configuration import (
    DeviceConfiguration,
    DeviceRuntimeConfiguration,
    InteractionPolicyConfiguration,
    InteractionRuntimeConfiguration,
    ScrollInteractionPolicy,
    SwipeInteractionPolicy,
)
from fathom.schemas.remote import RemoteInteractionRequest
from fathom.schemas.results import ActionResult

logger = getLogger(__name__)


class ADBRemoteDeviceAdapter(DevicePort):
    """
    Adapter for controlling devices hosted on remote providers (e.g., Enricher).
    Implements the standard Fathom Remote Device Protocol.
    """

    __ACTION_FAILURE_MESSAGE = "Failed to execute the action on the remote device. Please retry."

    def __init__(self, configuration: DeviceConfiguration) -> None:
        """
        Initialize remote device adapter.
        """

        remote = configuration.remote

        if not remote.provider_url or not remote.session_id:
            raise PortError("Remote device requires remote.provider_url and remote.session_id")

        self.__session = remote.session_id
        self.__execution_id = remote.execution_id

        self.__token = remote.authentication_token
        self.__url = remote.provider_url.rstrip("/") + "/"

        interaction = (
            configuration.ios.interaction
            if configuration.platform == DevicePlatform.IOS
            else configuration.android.interaction
        )

        self.__runtime_configuration = DeviceRuntimeConfiguration(
            identifier=self.__session,
            platform=configuration.platform,
            interaction=InteractionRuntimeConfiguration(
                policy=InteractionPolicyConfiguration(
                    swipe=SwipeInteractionPolicy(
                        duration=interaction.policy.swipe.duration,
                        edge_margin_ratio=interaction.policy.swipe.edge_margin_ratio,
                        minimum_edge_margin=interaction.policy.swipe.minimum_edge_margin,
                        maximum_edge_margin=interaction.policy.swipe.maximum_edge_margin,
                    ),
                    scroll=ScrollInteractionPolicy(
                        edge_margin_ratio=interaction.policy.scroll.edge_margin_ratio,
                        minimum_edge_margin=interaction.policy.scroll.minimum_edge_margin,
                        maximum_edge_margin=interaction.policy.scroll.maximum_edge_margin,
                    ),
                )
            ),
        )
        base_url = urljoin(self.__url, f"sessions/{self.__session}/interaction/")

        self.__client = httpx.AsyncClient(
            http2=True,
            base_url=base_url,
            timeout=REMOTE_DEVICE_REQUEST_TIMEOUT_SECONDS,
            headers={"Authorization": f"Bearer {self.__token}"} if self.__token else {},
        )

        self.__cached_dimensions: Optional[Tuple[int, int]] = None

    @retry(  # type: ignore[untyped-decorator]
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception(DeviceError.is_transient),
        before_sleep=before_sleep_log(logger, logging.WARNING, exc_info=True),
    )
    async def __execute_request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        """
        Executes an HTTP request with automatic retries for transient errors (5xx, timeouts). Immediately raises on 4xx errors.
        """

        if self.__client.is_closed:
            logger.warning(
                "Remote device request skipped because the HTTP client is already closed."
            )
            raise DeviceConnectionClosedError(
                "Remote device request attempted after the HTTP client was already closed.",
            )

        try:
            response = await self.__client.request(method, path, **kwargs)
        except RuntimeError as exception:
            if self.__is_closed_client_error(exception=exception):
                logger.exception(
                    "Remote device request failed because the HTTP client was closed during request execution."
                )
                raise DeviceConnectionClosedError(
                    "Remote device request failed because the HTTP client closed during request execution."
                ) from exception
            raise

        response.raise_for_status()

        return response

    @property
    def configuration(self) -> DeviceRuntimeConfiguration:
        """
        Returns platform-neutral runtime configuration.
        """

        return self.__runtime_configuration

    def __is_closed_client_error(self, *, exception: RuntimeError) -> bool:
        """
        Determine whether a request failed because the underlying client was closed.
        """

        return self.__client.is_closed or "client has been closed" in str(exception).lower()

    async def get_snapshot(self) -> Tuple[bytes, Optional[str]]:
        """
        Retrieve atomic snapshot (Screenshot + XML) from remote provider.
        """

        try:
            params = {"execution_id": self.__execution_id} if self.__execution_id else {}
            response = await self.__execute_request("POST", "snapshot", params=params)

            data = response.content

            if len(data) < 4:
                raise DeviceError("Snapshot response too short for header")

            # Binary Unpacking: [4b image_length][image][xml]
            header = data[:4]
            image_length = struct.unpack("!I", header)[0]

            image_end = 4 + image_length
            if len(data) < image_end:
                raise DeviceError("Snapshot response truncated (image)")

            image_bytes = data[4:image_end]
            xml_bytes = data[image_end:]

            image_header = image_bytes[:20].hex() if image_bytes else "EMPTY"
            logger.info(
                f"Received snapshot: Img={len(image_bytes)}b XML={len(xml_bytes)}b Header={image_header}"
            )

            return image_bytes, xml_bytes.decode("utf-8", errors="ignore")

        except httpx.HTTPStatusError as exception:
            # Re-wrap without losing original trace. HTTPStatusError is a subclass of HTTPError
            status = exception.response.status_code
            logger.exception("Remote snapshot request failed with HTTP %s.", status)
            raise DeviceError(f"Remote snapshot request failed with HTTP {status}.") from exception

        except httpx.HTTPError as exception:
            logger.exception("Remote snapshot request failed due to a transport error.")
            raise DeviceError(
                f"Remote snapshot request failed due to a transport error: {exception}"
            ) from exception

        except DeviceError:
            raise

        except Exception as exception:
            logger.exception("Remote snapshot response could not be parsed.")
            raise DeviceError(
                f"Remote snapshot response could not be parsed: {exception}"
            ) from exception

    async def tap(self, *, x: int, y: int) -> ActionResult:
        """
        Execute remote tap.
        """

        request = RemoteInteractionRequest(
            action=InteractionAction.TAP, x=x, y=y, execution_id=self.__execution_id
        )
        return await self.__send_command(request)

    async def type(self, *, text: str) -> ActionResult:
        """
        Execute remote text input.
        """

        request = RemoteInteractionRequest(
            action=InteractionAction.TYPE, text=text, execution_id=self.__execution_id
        )
        return await self.__send_command(request)

    async def swipe(
        self,
        *,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        duration: Optional[int] = None,
        speed: Optional[SwipeSpeed] = None,
    ) -> ActionResult:
        """
        Execute remote swipe.
        """

        runtime_configuration = self.configuration
        resolved_duration = duration or 300

        if duration is None and runtime_configuration:
            resolved_duration = runtime_configuration.interaction.policy.swipe.duration

        request = RemoteInteractionRequest(
            speed=speed,
            duration=resolved_duration,
            points=[x1, y1, x2, y2],
            action=InteractionAction.SWIPE,
            execution_id=self.__execution_id,
        )
        return await self.__send_command(request)

    async def back(self) -> ActionResult:
        """
        Execute remote back press.
        """

        request = RemoteInteractionRequest(
            action=InteractionAction.BACK, execution_id=self.__execution_id
        )
        return await self.__send_command(request)

    async def home(self) -> ActionResult:
        """
        Execute remote home press.
        """

        request = RemoteInteractionRequest(
            action=InteractionAction.HOME, execution_id=self.__execution_id
        )
        return await self.__send_command(request)

    async def get_dimensions(self) -> Tuple[int, int]:
        """
        Get screen dimensions.
        """

        if self.__cached_dimensions:
            return self.__cached_dimensions

        request = RemoteInteractionRequest(
            action=InteractionAction.GET_DIMENSIONS, execution_id=self.__execution_id
        )

        try:
            response = await self.__execute_request(
                "POST", "action", json=request.model_dump(exclude_none=True)
            )

            payload = self.__parse_response(response.json())

            width = payload.get("width")
            height = payload.get("height")

            if width and height:
                self.__cached_dimensions = (int(width), int(height))
                return self.__cached_dimensions

            raise DeviceError("Get dimensions: Response missing width or height fields")

        except httpx.HTTPError as exception:
            logger.exception("Remote dimensions request failed.")
            raise DeviceError(f"Remote dimensions request failed: {exception}") from exception

        except DeviceError:
            raise

        except Exception as exception:
            logger.exception("Remote dimensions response could not be parsed.")
            raise DeviceError(
                f"Remote dimensions response could not be parsed: {exception}"
            ) from exception

    async def capture_screen(self) -> bytes:
        """
        Capture device screenshot.
        """

        request = RemoteInteractionRequest(
            action=InteractionAction.GET_SCREENSHOT, execution_id=self.__execution_id
        )

        try:
            response = await self.__execute_request(
                "POST", "action", json=request.model_dump(exclude_none=True)
            )

            payload = self.__parse_response(response.json())

            if buffer := payload.get("base64"):
                return base64.b64decode(buffer)

            raise DeviceError("Capture screen: No base64 data in screenshot response")

        except httpx.HTTPError as exception:
            logger.exception("Remote screenshot request failed.")
            raise DeviceError(f"Remote screenshot request failed: {exception}") from exception

        except DeviceError:
            raise

        except Exception as exception:
            logger.exception("Remote screenshot response could not be decoded.")
            raise DeviceError(
                f"Remote screenshot response could not be decoded: {exception}"
            ) from exception

    async def dump_hierarchy(self) -> Optional[str]:
        """
        Dump UI hierarchy to XML string.
        """

        request = RemoteInteractionRequest(
            action=InteractionAction.GET_XML, execution_id=self.__execution_id
        )

        try:
            response = await self.__execute_request(
                "POST", "action", json=request.model_dump(exclude_none=True)
            )

            payload = self.__parse_response(response.json())

            xml_content = payload.get("xml")
            return str(xml_content) if xml_content is not None else None

        except httpx.HTTPError as exception:
            logger.exception("Remote hierarchy request failed.")
            raise DeviceError(f"Remote hierarchy request failed: {exception}") from exception

        except DeviceError:
            raise

        except Exception as exception:
            logger.exception("Remote hierarchy response could not be parsed.")
            raise DeviceError(
                f"Remote hierarchy response could not be parsed: {exception}"
            ) from exception

    async def get_current_package(self) -> str:
        """
        Get current package.
        """

        request = RemoteInteractionRequest(
            action=InteractionAction.GET_CURRENT_PACKAGE, execution_id=self.__execution_id
        )

        try:
            response = await self.__execute_request(
                "POST", "action", json=request.model_dump(exclude_none=True)
            )

            data = response.json()
            logger.info(f"Response of current package command: {data}")

            payload = self.__parse_response(data)
            package_name = payload.get("package")
            package = str(package_name) if package_name else "unknown"
            return package

        except httpx.HTTPError as exception:
            logger.exception("Remote foreground-package request failed.")
            raise DeviceError(
                f"Remote foreground-package request failed: {exception}"
            ) from exception

        except DeviceError:
            raise

        except Exception as exception:
            logger.exception("Remote foreground-package response could not be parsed.")
            raise DeviceError(
                f"Remote foreground-package response could not be parsed: {exception}"
            ) from exception

    async def wait_for_device(self, *, timeout: float) -> bool:
        """
        Wait for device to be ready.
        """

        start = time.time()
        while (time.time() - start) < timeout:
            try:
                _ = await self.__execute_request("GET", "")
                return True
            except httpx.HTTPError:
                await asyncio.sleep(1.0)

        return False

    def __parse_response(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Standardized JSend payload extractor.
        """

        # 1. Check for standard JSend 'data' field
        if "data" in response_data and isinstance(response_data["data"], dict):
            return cast("Dict[str, Any]", response_data["data"])

        # 2. Fallback to legacy 'content' field
        if "content" in response_data and isinstance(response_data["content"], dict):
            return cast("Dict[str, Any]", response_data["content"])

        # 3. Return the root if neither is present (direct response)
        return response_data

    async def __send_command(self, request: RemoteInteractionRequest) -> ActionResult:
        """
        Helper to transmit interaction request.
        """

        start = time.time()

        try:
            response = await self.__execute_request(
                "POST", "action", json=request.model_dump(exclude_none=True)
            )

            data = response.json()
            success = data.get("status") != "ERROR"
            error = data.get("message") if not success else None

            return ActionResult(
                error=error,
                success=success,
                duration=int((time.time() - start) * 1000),
            )
        except Exception as exception:
            logger.exception("Remote command failed.", stack_info=True)
            message = (
                exception.display(fallback=self.__ACTION_FAILURE_MESSAGE)
                if isinstance(exception, FathomError) and hasattr(exception, "display")
                else self.__ACTION_FAILURE_MESSAGE
            )
            return ActionResult(
                success=False,
                error=message,
                duration=int((time.time() - start) * 1000),
            )

    async def close(self) -> None:
        """
        Cleanup HTTP client.
        """

        await self.__client.aclose()
