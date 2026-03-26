from __future__ import annotations

from datetime import datetime
from logging import getLogger
from pathlib import Path
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
        Saves artifact data locally with a structured path.
        """

        try:
            if not metadata:
                raise ValueError("Storage metadata is required for saving artifacts")

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            session = metadata.get("session_id")
            package = metadata.get("package_name")
            # Activity name is optional, fallback to package if missing
            activity = metadata.get("activity_name") or package
            filename_meta = metadata.get("filename")
            category = self.__resolve_category(metadata=metadata)

            if not all([session, package]):
                raise ValueError(f"Missing required artifact metadata: {session=}, {package=}")

            # Ensure types for mypy after validation
            package_str = str(package)
            session_str = str(session)
            activity_str = str(activity)

            # Sanitize components to be safe for filenames
            package = "".join(char for char in package_str if char.isalnum() or char in "._-")
            session = "".join(char for char in session_str if char.isalnum() or char in "._-")
            activity = "".join(char for char in activity_str if char.isalnum() or char in "._-")

            filename = (
                filename_meta
                or f"{timestamp}__{activity}{self.__resolve_extension(category=category)}"
            )

            path = self.__resolve_destination_path(
                category=category, filename=filename, session_id=session, package_name=package
            )

            self.__write(path=path, data=data)
            logger.debug("Saved local %s artifact: %s", category, path)
            return str(path.absolute())
        except Exception as exception:
            logger.warning("Failed to save local artifact: %s", exception)
            raise VisionError(f"Local save failed: {exception}") from exception

    def __write(self, path: Path, data: bytes) -> None:
        """
        Handles physical write to disk.
        """

        with path.open("wb") as handle:
            handle.write(data)

    def __resolve_category(self, *, metadata: Dict[str, Any]) -> str:
        """
        Resolve a canonical artifact category from storage metadata.
        """

        raw_category = metadata.get("category") or metadata.get("type") or "screenshot"
        category = str(raw_category).strip().lower()

        if category in {"screenshot", "screenshots"}:
            return "screenshot"

        if category == "annotated":
            return "annotated"

        if category == "xmls":
            return "xmls"

        if category == "traces":
            return "traces"

        if category == "history":
            return "history"

        return "screenshot"

    def __resolve_extension(self, *, category: str) -> str:
        """
        Resolve the default extension for a category when filename is not provided.
        """

        if category == "xmls":
            return ".xml"

        if category == "history":
            return ".txt"

        return ".png"

    def __resolve_destination_path(
        self, *, category: str, filename: str, session_id: str, package_name: str
    ) -> Path:
        """
        Resolve the filesystem path for the artifact category.
        """

        if category == "annotated":
            return self.__path_manager.get_annotated_path(
                filename=filename, session_id=session_id, package_name=package_name
            )

        if category == "xmls":
            return self.__path_manager.get_xml_path(
                filename=filename, session_id=session_id, package_name=package_name
            )

        if category == "traces":
            return self.__path_manager.get_trace_path(
                filename=filename, session_id=session_id, package_name=package_name
            )
        if category == "history":
            return (
                self.__path_manager.get_history_directory(
                    session_id=session_id,
                    package_name=package_name,
                )
                / filename
            )

        return self.__path_manager.get_screenshot_path(
            filename=filename,
            session_id=session_id,
            package_name=package_name,
        )
