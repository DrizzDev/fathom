from __future__ import annotations

from datetime import datetime
from logging import getLogger
from typing import TYPE_CHECKING, Any, Dict, Optional

from fathom.core.exceptions import VisionError
from fathom.interfaces import IImageStorage

if TYPE_CHECKING:
    from pathlib import Path

    from fathom.base.paths import SharedPathManager

logger = getLogger(__name__)


class LocalImageStorage(IImageStorage):
    """
    Handles saving images to the local filesystem.
    """

    def __init__(self, path_manager: SharedPathManager) -> None:
        self.__path_manager = path_manager

    async def save(self, data: bytes, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Saves image data locally with a structured path.
        """

        try:
            if not metadata:
                raise ValueError("Storage metadata is required for saving screenshots")

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            session = metadata.get("session_id")
            package = metadata.get("package_name")
            # Activity name is optional, fallback to package if missing
            activity = metadata.get("activity_name") or package
            filename_meta = metadata.get("filename")

            if not all([session, package]):
                raise ValueError(f"Missing required screenshot metadata: {session=}, {package=}")

            # Ensure types for mypy after validation
            package_str = str(package)
            session_str = str(session)
            activity_str = str(activity)

            # Sanitize components to be safe for filenames
            package = "".join(char for char in package_str if char.isalnum() or char in "._-")
            session = "".join(char for char in session_str if char.isalnum() or char in "._-")
            activity = "".join(char for char in activity_str if char.isalnum() or char in "._-")

            # Use explicit filename if provided (e.g. for history artifacts), otherwise generate one
            filename = filename_meta or f"{timestamp}__{activity}.png"

            # Use path manager to enforce unified structure
            path = self.__path_manager.get_screenshot_path(
                package_name=package, session_id=session, filename=filename
            )

            self.__write(path=path, data=data)
            logger.debug(f"Saved local screenshot: {path}")
            return str(path.absolute())
        except Exception as exception:
            logger.warning(f"Failed to save local screenshot: {exception}")
            raise VisionError(f"Local save failed: {exception}") from exception

    def __write(self, path: Path, data: bytes) -> None:
        """
        Handles physical write to disk.
        """

        with path.open("wb") as handle:
            handle.write(data)
