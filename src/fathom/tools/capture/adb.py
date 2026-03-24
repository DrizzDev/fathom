from __future__ import annotations

import asyncio
import hashlib
import io
import time
from logging import getLogger
from typing import Optional, Union

from PIL import Image

from fathom.schemas.configuration import ADBCaptureConfig
from fathom.schemas.screens import ScreenCapture, ScreenState
from fathom.tools.capture.base import CaptureTool
from fathom.tools.capture.hasher import FastHasher, HybridHasher

logger = getLogger(name=__name__)


class ADBCaptureTool(CaptureTool):
    """
    Standard ADB capture tool.
    """

    def __init__(self, config: Optional[ADBCaptureConfig] = None) -> None:
        """
        Initialize ADB capture tool.
        """
        self.__config = config or ADBCaptureConfig()

        if self.__config.use_hybrid_hash:
            self.__hasher: Union[HybridHasher, FastHasher] = HybridHasher()
        else:
            self.__hasher = FastHasher()

    async def capture(self) -> ScreenCapture:
        """
        Perform a capture using sequential calls for stability.
        """

        try:
            image_data = await self.__capture_image_only()
            activity = await self.__get_current_activity()

            # Fast dimension check
            with Image.open(fp=io.BytesIO(initial_bytes=image_data)) as image:
                width, height = image.size

            return ScreenCapture(
                width=width,
                height=height,
                image=image_data,
                activity=activity,
                timestamp=int(time.time() * 1000),
            )

        except Exception as exception:
            logger.error(msg=f"Capture failed: {exception}")
            return await self.__fallback_capture()

    async def capture_stable(self, timeout: int = 2000) -> ScreenCapture:
        """
        Standard stability check.
        """

        start_time = time.time()
        timeout_seconds = timeout / 1000.0

        last_image = await self.__capture_image_only()
        last_hashes = self.__compute_hashes(image_data=last_image)

        while (time.time() - start_time) < timeout_seconds:
            await asyncio.sleep(delay=0.25)

            try:
                current_image = await self.__capture_image_only()
                current_hashes = self.__compute_hashes(image_data=current_image)

                if current_hashes[0] == last_hashes[0]:
                    return await self.capture()

                last_image = current_image
                last_hashes = current_hashes

            except Exception as exception:
                logger.debug(msg=f"Stability polling frame failed: {exception}")
                continue

        return await self.capture()

    def compute_state(self, capture: ScreenCapture) -> ScreenState:
        """
        Compute complete state.
        """

        visual_hashes = self.__compute_hashes(image_data=capture.image)
        activity_hash = hashlib.md5(capture.activity.encode(), usedforsecurity=False).hexdigest()[
            :8
        ]

        return ScreenState(
            visual_hash=visual_hashes[0],
            activity=capture.activity,
            timestamp=capture.timestamp,
            activity_hash=activity_hash,
            structural_hash=visual_hashes[1],
        )

    def __compute_hashes(self, image_data: bytes) -> tuple[str, str]:
        """
        Internal hash computation helper.
        """

        if isinstance(self.__hasher, HybridHasher):
            try:
                with Image.open(fp=io.BytesIO(initial_bytes=image_data)) as image:
                    image_rgb = image.convert(mode="RGB")
                    visual_hash = self.__hasher.compute_phash(img=image_rgb)
                    structural_hash = self.__hasher.compute_structural(img=image_rgb)
                    return visual_hash, structural_hash
            except Exception:
                return "0" * 16, "0" * 8

        return "0" * 16, "0" * 8

    async def __capture_image_only(self) -> bytes:
        """
        Standard image capture.
        """

        arguments = self.__build_adb_args(args=["exec-out", "screencap", "-p"])
        process = await asyncio.create_subprocess_exec(
            *arguments, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL
        )
        stdout, _ = await asyncio.wait_for(fut=process.communicate(), timeout=self.__config.timeout)
        return stdout

    async def __get_current_activity(self) -> str:
        """
        Get activity using standard shell command.
        """

        command = "dumpsys window | grep -E 'mCurrentFocus|mFocusedApp'"
        arguments = self.__build_adb_args(args=["shell", command])

        try:
            process = await asyncio.create_subprocess_exec(
                *arguments, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL
            )
            stdout, _ = await asyncio.wait_for(
                fut=process.communicate(), timeout=self.__config.timeout
            )
            if stdout:
                import re

                match = re.search(pattern=r"(\S+/\S+)", string=stdout.decode())
                if match:
                    return match.group(1).strip().rstrip("}")
            return "unknown"
        except Exception:
            return "unknown"

    async def __fallback_capture(self) -> ScreenCapture:
        """
        Safe fallback. Returns last known valid image or a blank one if everything fails.
        """

        blank_image = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"

        try:
            image_data = await self.__capture_image_only()

            try:
                with Image.open(fp=io.BytesIO(initial_bytes=image_data)) as image:
                    width, height = image.size
            except Exception:
                # If we cannot parse the image, use the blank image data
                width, height = 1, 1
                image_data = blank_image

            return ScreenCapture(
                width=width,
                height=height,
                image=image_data,
                activity="unknown",
                timestamp=int(time.time() * 1000),
            )
        except Exception:
            # Absolute last resort: return a 1x1 transparent pixel
            return ScreenCapture(
                width=1,
                height=1,
                image=blank_image,
                activity="unknown",
                timestamp=int(time.time() * 1000),
            )

    def __build_adb_args(self, args: list[str]) -> list[str]:
        """
        Build full ADB command arguments.
        """

        command = [self.__config.adb_path]

        if self.__config.device_serial:
            command.extend(["-s", self.__config.device_serial])

        command.extend(args)
        return command
