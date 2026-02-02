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

    Example:
        ```python
        capture = ADBCaptureTool(ADBCaptureConfig(
            device_serial="emulator-5554"
        ))
        screenshot = await capture.capture()
        state = capture.compute_state(screenshot)
        ```
    """

    def __init__(self, config: Optional[ADBCaptureConfig] = None) -> None:
        """
        Initialize ADB capture tool.

        Args:
            config: ADB capture configuration.
        """

        self.__config = config or ADBCaptureConfig()

        if self.__config.use_hybrid_hash:
            self.__hasher: Union[HybridHasher, FastHasher] = HybridHasher()
        else:
            self.__hasher = FastHasher()

    async def capture(self) -> ScreenCapture:
        """
        Capture screenshot from device.

        Returns:
            ScreenCapture with image bytes and metadata.
        """

        image = await self.__capture_screenshot()
        activity = await self.__get_current_activity()

        timestamp = int(time.time() * 1000)

        # Get dimensions
        try:
            import io

            from PIL import Image

            with Image.open(io.BytesIO(image)) as img:
                width, height = img.size
        except ImportError as exception:
            width, height = 1080, 1920
            logger.warning(
                "Pillow not installed, using fallback dimensions",
                extra={"exception": str(exception)},
            )

        return ScreenCapture(
            image=image,
            width=width,
            height=height,
            activity=activity,
            timestamp=timestamp,
        )

    async def capture_stable(self, timeout: int = 2000) -> ScreenCapture:
        """
        Capture screen after waiting for stability.

        Args:
            timeout: Maximum wait time in milliseconds.

        Returns:
            Stable screen capture.
        """

        start_time = time.time()
        timeout_sec = timeout / 1000.0

        # Initial capture
        last_capture = await self.capture()
        last_state = self.compute_state(last_capture)

        while (time.time() - start_time) < timeout_sec:
            await asyncio.sleep(0.2)  # 200ms interval

            try:
                current_capture = await self.capture()
                current_state = self.compute_state(current_capture)

                if current_state.is_same_screen(last_state):
                    return current_capture

                last_capture = current_capture
                last_state = current_state

            except Exception as exception:
                logger.warning(
                    "Transient capture error during stability check",
                    extra={"error": str(exception)},
                )
                continue

        return last_capture

    def compute_state(self, capture: ScreenCapture) -> ScreenState:
        """
        Compute state from screen capture.

        Args:
            capture: Screen capture.

        Returns:
            ScreenState with hashes and metadata.
        """

        if isinstance(self.__hasher, HybridHasher):
            try:
                with Image.open(io.BytesIO(capture.image)) as img:
                    img = img.convert("RGB")
                    visual_hash = self.__hasher.compute_phash(img)
                    structural_hash = self.__hasher.compute_structural(img)
            except ImportError:
                visual_hash = "0" * 16
                structural_hash = "0" * 8
        else:
            # FastHasher fallback
            visual_hash = "0" * 16
            structural_hash = "0" * 8

        activity_hash = hashlib.md5(capture.activity.encode()).hexdigest()[:8]  # nosec

        return ScreenState(
            visual_hash=visual_hash,
            activity=capture.activity,
            timestamp=capture.timestamp,
            activity_hash=activity_hash,
            structural_hash=structural_hash,
        )

    async def __capture_screenshot(self) -> bytes:
        """
        Capture screenshot via ADB.

        Returns:
            PNG image bytes.
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
                error = stderr.decode().strip() if stderr else "Screenshot failed"
                raise RuntimeError(f"ADB screenshot failed: {error}")

            if not stdout:
                raise RuntimeError("Empty screenshot from ADB")

            return stdout

        except asyncio.TimeoutError as err:
            raise RuntimeError("Screenshot timed out") from err

    async def __get_current_activity(self) -> str:
        """
        Get current activity name.

        Returns:
            Activity name or "unknown".
        """

        args = self.__build_adb_args(
            [
                "shell",
                "dumpsys activity activities | grep mResumedActivity",
            ]
        )

        try:
            process = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, _ = await asyncio.wait_for(
                process.communicate(),
                timeout=5.0,
            )

            if stdout:
                output = stdout.decode()
                import re

                match = re.search(r"(\S+/\S+)", output)
                if match:
                    return match.group(1)

            return "unknown"

        except Exception:
            return "unknown"

    def __build_adb_args(self, args: list[str]) -> list[str]:
        """
        Build full ADB command arguments.

        Args:
            args: ADB subcommand and arguments.

        Returns:
            Full command list.
        """

        cmd = [self.__config.adb_path]
        if self.__config.device_serial:
            cmd.extend(["-s", self.__config.device_serial])

        cmd.extend(args)
        return cmd
