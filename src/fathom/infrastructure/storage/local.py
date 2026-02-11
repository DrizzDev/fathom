from __future__ import annotations

from datetime import datetime
from logging import getLogger
from pathlib import Path
from typing import Any, Dict, Optional

from fathom.exceptions import VisionError
from fathom.interfaces import IImageStorage

logger = getLogger(__name__)


class LocalImageStorage(IImageStorage):
    """
    Handles saving images to the local filesystem.
    """

    async def save(self, data: bytes, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Saves image data locally with a structured path.
        Structure: assets/screenshot/YYYY-MM-DD/{package}/{session}/{timestamp}__{activity}.png
        """

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = self.__build_path(timestamp=timestamp, metadata=metadata)
            self.__write(path=path, data=data)
            logger.debug(f"Saved local screenshot: {path}")
            return str(path.absolute())
        except Exception as exception:
            logger.warning(f"Failed to save local screenshot: {exception}")
            raise VisionError(f"Local save failed: {exception}") from exception

    def __build_path(self, timestamp: str, metadata: Optional[Dict[str, Any]]) -> Path:
        """
        Constructs the destination path for the screenshot using metadata.
        """

        base = Path("assets/screenshot")
        folder = datetime.now().strftime("%Y-%m-%d")

        package = "unknown_app"
        session = "default_session"
        activity = "unknown_screen"

        if metadata:
            session = metadata.get("session_id") or session
            package = metadata.get("package_name") or package
            activity = metadata.get("activity_name") or activity

        # Sanitize components to be safe for filenames
        package = "".join(char for char in package if char.isalnum() or char in "._-")
        session = "".join(char for char in session if char.isalnum() or char in "._-")
        activity = "".join(char for char in activity if char.isalnum() or char in "._-")

        directory = base / folder / package / session
        directory.mkdir(parents=True, exist_ok=True)

        return directory / f"{timestamp}__{activity}.png"

    def __write(self, path: Path, data: bytes) -> None:
        """
        Handles physical write to disk.
        """

        with path.open("wb") as handle:
            handle.write(data)
