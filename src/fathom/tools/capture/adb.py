import asyncio
import hashlib
import io
import time
from dataclasses import dataclass
from logging import getLogger
from typing import Optional, Union

from PIL import Image

from fathom.schemas.screens import ScreenCapture, ScreenState
from fathom.tools.capture.base import CaptureTool
from fathom.tools.capture.hasher import FastHasher, HybridHasher

logger = getLogger(__name__)


@dataclass(frozen=True)
class ADBCaptureConfig:
    """
    Configuration for ADB capture tool.
    """

    adb_path: str = "adb"
    timeout: float = 10.0
    use_hybrid_hash: bool = True
    device_serial: Optional[str] = None


class ADBCaptureTool(CaptureTool):
    """
    Capture tool using ADB for real device screenshots.
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
        Capture screenshot from device.
        """
        image = await self.__capture_screenshot()
        activity = await self.__get_current_activity()

        timestamp = int(time.time() * 1000)

        # Get dimensions
        try:
            with Image.open(io.BytesIO(image)) as img:
                width, height = img.size
        except Exception as exception:
            width, height = 1080, 1920
            logger.warning(f"Fallback dimensions used: {exception}")

        return ScreenCapture(
            image=image,
            width=width,
            height=height,
            activity=activity,
            timestamp=timestamp,
        )

    async def capture_stable(self, timeout: int = 2000) -> ScreenCapture:
        """
        Capture screen after waiting for stability with reduced overhead.
        """
        start_time = time.time()
        timeout_sec = timeout / 1000.0

        last_capture = await self.capture()
        last_state = self.compute_state(last_capture)

        while (time.time() - start_time) < timeout_sec:
            await asyncio.sleep(0.2)

            try:
                current_capture = await self.capture()
                current_state = self.compute_state(current_capture)

                if current_state.is_same_screen(last_state):
                    return current_capture

                last_capture = current_capture
                last_state = current_state

            except Exception as exception:
                logger.debug(f"Stability check error: {exception}")
                continue

        return last_capture

    def compute_state(self, capture: ScreenCapture) -> ScreenState:
        """
        Compute state from screen capture.
        """
        if isinstance(self.__hasher, HybridHasher):
            try:
                with Image.open(io.BytesIO(capture.image)) as img:
                    img = img.convert("RGB")
                    visual_hash = self.__hasher.compute_phash(img)
                    structural_hash = self.__hasher.compute_structural(img)
            except Exception:
                visual_hash = "0" * 16
                structural_hash = "0" * 8
        else:
            visual_hash = "0" * 16
            structural_hash = "0" * 8

        activity_hash = hashlib.md5(capture.activity.encode(), usedforsecurity=False).hexdigest()[
            :8
        ]

        return ScreenState(
            visual_hash=visual_hash,
            activity=capture.activity,
            timestamp=capture.timestamp,
            activity_hash=activity_hash,
            structural_hash=structural_hash,
        )

    async def __capture_screenshot(self) -> bytes:
        """
        Capture screenshot via ADB exec-out.
        """
        args = self.__build_adb_args(["exec-out", "screencap", "-p"])

        try:
            process = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.__config.timeout,
            )

            if process.returncode != 0:
                raise RuntimeError(f"ADB screenshot failed: {stderr.decode()}")

            return stdout

        except asyncio.TimeoutError as err:
            raise RuntimeError("Screenshot timed out") from err

    async def __get_current_activity(self) -> str:
        """
        Get current activity name using robust multi-fallback detection.
        """
        # Combined command to reduce round-trips
        cmd = "dumpsys activity activities | grep -E 'mResumedActivity' || dumpsys window | grep -E 'mCurrentFocus'"
        args = self.__build_adb_args(["shell", cmd])

        try:
            process = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, _ = await asyncio.wait_for(
                process.communicate(),
                timeout=3.0,
            )

            if stdout:
                output = stdout.decode()
                import re

                # Try ActivityRecord pattern
                match = re.search(r"ActivityRecord\{.*?\s(\S+)\s+t\d+\}", output)
                if match:
                    return match.group(1)

                # Try mCurrentFocus pattern
                match = re.search(r"mCurrentFocus=Window\{.*?\s(\S+)\}", output)
                if match:
                    val = match.group(1)
                    if "/" in val:
                        pkg, cls = val.split("/", 1)
                        if cls.startswith(pkg):
                            return f"{pkg}/.{cls[len(pkg) + 1 :]}"
                    return val

            return "unknown"

        except Exception as exception:
            logger.debug(f"Activity detection failed: {exception}")
            return "unknown"

    def __build_adb_args(self, args: list[str]) -> list[str]:
        """
        Build full ADB command arguments.
        """
        cmd = [self.__config.adb_path]
        if self.__config.device_serial:
            cmd.extend(["-s", self.__config.device_serial])

        cmd.extend(args)
        return cmd
