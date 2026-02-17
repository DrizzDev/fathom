"""
Service for perceiving device state.
Handles screen capture, persistence, and visual hashing.
"""

from __future__ import annotations

import hashlib
import time
from logging import getLogger

from fathom.core.exceptions import PortError
from fathom.interfaces.device import DevicePort
from fathom.interfaces.storage import StoragePort
from fathom.interfaces.telemetry import TelemetryPort
from fathom.schemas.screens import ScreenCapture

logger = getLogger(__name__)


class PerceptionService:
    """
    Perception service for capturing and hashing screen state.
    """

    def __init__(
        self,
        device: DevicePort,
        storage: StoragePort,
        telemetry: TelemetryPort,
    ) -> None:
        self.__device = device
        self.__storage = storage
        self.__telemetry = telemetry

    async def perceive(self) -> ScreenCapture:
        """
        Capture current screen state via DevicePort.
        
        Returns:
            ScreenCapture with screenshot data
        """
        screenshot_bytes = await self.__device.capture_screen()

        # Get screen dimensions
        width, height = await self.__device.get_screen_size()

        # Get current activity
        try:
            activity = await self.__device.get_current_package()
        except Exception as exception:
            self.__telemetry.warning(
                "Failed to get current package",
                error=str(exception),
            )
            activity = "unknown"

        # Store screenshot artifact
        storage_id = await self.__persist_capture(data=screenshot_bytes)

        return ScreenCapture(
            width=width,
            height=height,
            activity=activity,
            image=screenshot_bytes,
            timestamp=int(time.time() * 1000),
            metadata={"storage_id": storage_id},
        )

    async def __persist_capture(self, data: bytes) -> str:
        """Persists screenshot to storage."""
        return await self.__storage.save(
            data=data,
            metadata={"type": "screenshot", "timestamp": time.time()},
        )

    def compute_visual_hash(self, capture: ScreenCapture) -> str:
        """
        Compute visual hash for screen capture.
        """
        from fathom.constants.execution import VISUAL_HASH_LENGTH

        return hashlib.sha256(capture.image).hexdigest()[:VISUAL_HASH_LENGTH]
