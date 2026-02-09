from __future__ import annotations

from datetime import datetime
from logging import getLogger
from pathlib import Path

from fathom.exceptions import VisionError
from fathom.interfaces import IImageStorage

logger = getLogger(__name__)


class LocalImageStorage(IImageStorage):
    """
    Handles saving images to the local filesystem.
    """

    async def save(self, data: bytes) -> str:
        """
        Saves image data locally with a human-readable timestamp.
        """

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = self.__build_path(timestamp)
            self.__write(path=path, data=data)
            logger.debug(f"Saved local screenshot: {path}")
            return str(path.absolute())
        except Exception as exception:
            logger.warning(f"Failed to save local screenshot: {exception}")
            raise VisionError(f"Local save failed: {exception}") from exception

    def __build_path(self, timestamp: str) -> Path:
        """
        Constructs the destination path for the screenshot.
        """

        directory = Path("assets/screenshot")
        directory.mkdir(parents=True, exist_ok=True)

        return directory / f"{timestamp}.png"

    def __write(self, path: Path, data: bytes) -> None:
        """
        Handles physical write to disk.
        """

        with path.open("wb") as handle:
            handle.write(data)
