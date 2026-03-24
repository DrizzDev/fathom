from __future__ import annotations

from datetime import datetime
from logging import getLogger
from typing import TYPE_CHECKING, Any, Dict, Optional

from fathom.core.exceptions import VisionError
from fathom.infrastructure.storage.metadata import extract_metadata
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
            meta = extract_metadata(metadata or {})

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = meta.filename or f"{timestamp}__{meta.activity}.png"

            # Use path manager to enforce unified structure
            path = self.__path_manager.get_screenshot_path(
                package_name=meta.package, session_id=meta.session, filename=filename
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
