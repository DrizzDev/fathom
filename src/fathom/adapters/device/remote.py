from __future__ import annotations

import asyncio
import base64
import struct
import time
from logging import getLogger
from typing import Any, Dict, Optional, Tuple, cast

import httpx

from fathom.core.exceptions import DeviceError, PortError
from fathom.interfaces.device import DevicePort
from fathom.schemas.configuration import ADBConfiguration, DeviceConfiguration
from fathom.schemas.remote import RemoteInteractionRequest
from fathom.schemas.results import ActionResult

logger = getLogger(__name__)


class RemoteDeviceAdapter(DevicePort):
    """
    Adapter for controlling devices hosted on remote providers (e.g., Enricher).
    Implements the standard Fathom Remote Device Protocol.
    """

    def __init__(self, configuration: DeviceConfiguration) -> None:
        """
        Initialize remote device adapter.
        """

        if not configuration.provider_url or not configuration.session_id:
            raise PortError("Remote device requires provider_url and session_id")

        self.__session = configuration.session_id
        self.__execution_id = configuration.execution_id
        self.__token = configuration.authentication_token
        self.__url = configuration.provider_url.rstrip("/")

        self.__adb_config = ADBConfiguration(serial_number=self.__session)

        self.__client = httpx.AsyncClient(
            http2=True,
            timeout=30.0,
            base_url=f"{self.__url}/sessions/{self.__session}/interaction",
            headers={"Authorization": f"Bearer {self.__token}"} if self.__token else {},
        )

        self.__cached_dimensions: Optional[Tuple[int, int]] = None

    @property
    def configuration(self) -> ADBConfiguration:
        """
        Returns compatible ADB configuration.
        """

        return self.__adb_config

    async def get_snapshot(self) -> Tuple[bytes, Optional[str]]:
        """
        Retrieve atomic snapshot (Screenshot + XML) from remote provider.
        """

        try:
            params = {"execution_id": self.__execution_id} if self.__execution_id else {}
            response = await self.__client.post("/snapshot", params=params)
            response.raise_for_status()

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

        except httpx.HTTPError as exception:
            raise DeviceError(f"Remote snapshot failed: {exception}") from exception

        except DeviceError:
            raise

        except Exception as exception:
            raise DeviceError(f"Snapshot parsing error: {exception}") from exception

    async def tap(self, *, x: int, y: int) -> ActionResult:
        """
        Execute remote tap.
        """

        request = RemoteInteractionRequest(action="tap", x=x, y=y, execution_id=self.__execution_id)
        return await self.__send_command(request)

    async def type(self, *, text: str) -> ActionResult:
        """
        Execute remote text input.
        """

        request = RemoteInteractionRequest(
            action="type", text=text, execution_id=self.__execution_id
        )
        return await self.__send_command(request)

    async def swipe(
        self, *, x1: int, y1: int, x2: int, y2: int, duration: int = 300
    ) -> ActionResult:
        """
        Execute remote swipe.
        """

        request = RemoteInteractionRequest(
            action="swipe",
            points=[x1, y1, x2, y2],
            extras={"duration": duration},
            execution_id=self.__execution_id,
        )
        return await self.__send_command(request)

    async def back(self) -> ActionResult:
        """
        Execute remote back press.
        """

        request = RemoteInteractionRequest(action="back", execution_id=self.__execution_id)
        return await self.__send_command(request)

    async def home(self) -> ActionResult:
        """
        Execute remote home press.
        """

        request = RemoteInteractionRequest(action="home", execution_id=self.__execution_id)
        return await self.__send_command(request)

    async def get_dimensions(self) -> Tuple[int, int]:
        """
        Get screen dimensions.
        """

        if self.__cached_dimensions:
            return self.__cached_dimensions

        request = RemoteInteractionRequest(
            action="GET_DIMENSIONS", execution_id=self.__execution_id
        )

        try:
            response = await self.__client.post("/action", json=request.model_dump())
            response.raise_for_status()

            payload = self.__parse_response(response.json())

            width = payload.get("width")
            height = payload.get("height")

            if width and height:
                self.__cached_dimensions = (int(width), int(height))
                return self.__cached_dimensions

            raise DeviceError("Get dimensions: Response missing width or height fields")

        except httpx.HTTPError as exception:
            raise DeviceError(
                f"Get dimensions: Failed to fetch from remote: {exception}"
            ) from exception

        except DeviceError:
            raise

        except Exception as exception:
            raise DeviceError(
                f"Get dimensions: Failed to parse response: {exception}"
            ) from exception

    async def capture_screen(self) -> bytes:
        """
        Capture device screenshot.
        """

        request = RemoteInteractionRequest(
            action="GET_SCREENSHOT", execution_id=self.__execution_id
        )

        try:
            response = await self.__client.post("/action", json=request.model_dump())
            response.raise_for_status()

            payload = self.__parse_response(response.json())

            if buffer := payload.get("base64"):
                return base64.b64decode(buffer)

            raise DeviceError("Capture screen: No base64 data in screenshot response")

        except httpx.HTTPError as exception:
            raise DeviceError(
                f"Capture screen: Remote screenshot failed: {exception}"
            ) from exception

        except DeviceError:
            raise

        except Exception as exception:
            raise DeviceError(
                f"Capture screen: Failed to decode screenshot: {exception}"
            ) from exception

    async def dump_hierarchy(self) -> Optional[str]:
        """
        Dump UI hierarchy to XML string.
        """

        request = RemoteInteractionRequest(action="GET_XML", execution_id=self.__execution_id)

        try:
            response = await self.__client.post("/action", json=request.model_dump())
            response.raise_for_status()

            payload = self.__parse_response(response.json())

            xml_content = payload.get("xml")
            return str(xml_content) if xml_content is not None else None

        except httpx.HTTPError as exception:
            raise DeviceError(f"Dump hierarchy: Remote XML dump failed: {exception}") from exception

        except DeviceError:
            raise

        except Exception as exception:
            raise DeviceError(
                f"Dump hierarchy: Failed to parse XML response: {exception}"
            ) from exception

    async def get_current_package(self) -> str:
        """
        Get current package.
        """

        request = RemoteInteractionRequest(
            action="GET_CURRENT_PACKAGE", execution_id=self.__execution_id
        )

        try:
            response = await self.__client.post("/action", json=request.model_dump())
            response.raise_for_status()

            data = response.json()
            logger.info(f"Response of current package command: {data}")

            payload = self.__parse_response(data)
            package = payload.get("package", "unknown_app")

            return str(package)

        except httpx.HTTPError as exception:
            raise DeviceError(
                f"Get current package: Remote package check failed: {exception}"
            ) from exception

        except DeviceError:
            raise

        except Exception as exception:
            raise DeviceError(
                f"Get current package: Failed to parse package response: {exception}"
            ) from exception

    async def wait_for_device(self, *, timeout: float) -> bool:
        """
        Wait for device to be ready.
        """

        start = time.time()
        while (time.time() - start) < timeout:
            try:
                _ = await self.__client.get("/")
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
            response = await self.__client.post("/action", json=request.model_dump())
            response.raise_for_status()

            data = response.json()
            success = data.get("status") != "ERROR"
            error = data.get("message") if not success else None

            return ActionResult(
                error=error,
                success=success,
                duration=int((time.time() - start) * 1000),
            )
        except Exception as exception:
            logger.error(f"Remote command failed: {exception}")
            return ActionResult(
                success=False,
                error=str(exception),
                duration=int((time.time() - start) * 1000),
            )

    async def close(self) -> None:
        """
        Cleanup HTTP client.
        """

        await self.__client.aclose()
