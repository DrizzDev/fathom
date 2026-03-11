from __future__ import annotations

import asyncio
import contextlib
import tempfile
import xml.etree.ElementTree as ET  # nosec
from datetime import datetime
from logging import getLogger
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from fathom.base.paths import SharedPathManager
from fathom.constants import ActionType
from fathom.interfaces.storage import StoragePort
from fathom.processing.annotator import ImageAnnotator
from fathom.processing.drawer import BoundsGenerator
from fathom.schemas.screens import ScreenCapture
from fathom.schemas.ui import LabeledElement

logger = getLogger(__name__)


class HierarchyService:
    """
    Service responsible for UI hierarchy analysis. Optimized for high-speed grounding.
    """

    def __init__(self, storage: Optional[StoragePort] = None) -> None:
        """
        Initialize hierarchy service.
        """
        self.__storage = storage
        self.__label_map: Dict[str, Any] = {}
        self.__background_tasks: Set[asyncio.Task[Any]] = set()

    def __fire_and_forget(self, coroutine: Any) -> None:
        """
        Schedules a coroutine as a background task.
        """

        try:
            task = asyncio.create_task(coroutine)
            self.__background_tasks.add(task)
            task.add_done_callback(self.__background_tasks.discard)
        except Exception as exception:
            logger.warning(f"Failed to create background task: {exception}", stack_info=True)

    @property
    def label_map(self) -> Dict[str, Any]:
        """
        Returns current label mapping.
        """

        return self.__label_map.copy()

    async def process_xml_and_screen(
        self,
        xml: str,
        screen: ScreenCapture,
        *,
        session_id: str,
        package_name: str,
        path_manager: SharedPathManager,
        action_type: Optional[ActionType] = None,
    ) -> Tuple[Optional[ScreenCapture], Dict[str, Any]]:
        """
        Processes XML and screen data to identify UI elements.
        """

        start_time = datetime.now()
        xml_size_kb = len(xml.encode("utf-8")) / 1024

        logger.info(f"Processing hierarchy. XML Size: {xml_size_kb:.2f} KB, Action: {action_type}")

        if xml_size_kb < 0.2:
            logger.warning(f"XML too small ({xml_size_kb:.2f} KB), possibly invalid.")
            return screen, {}

        screenshot_path: Optional[Path] = None
        created_temporary_screenshot = False

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename_base = f"{timestamp}"

            screenshot_path, created_temporary_screenshot = self.__resolve_source_screenshot_path(
                screen=screen
            )

            # 1. Save Raw XML (using path manager)
            xml_path = path_manager.get_xml_path(
                package_name=package_name, session_id=session_id, filename=f"{filename_base}.xml"
            )
            xml_bytes = xml.encode("utf-8")
            self.__save_file(path=xml_path, data=xml_bytes, mode="wb")

            if self.__storage:
                self.__fire_and_forget(
                    self.__storage.save(
                        data=xml_bytes,
                        metadata={
                            "category": "xmls",
                            "session_id": session_id,
                            "package_name": package_name,
                            "filename": f"{filename_base}.xml",
                        },
                    )
                )

            # 2. Parse Elements
            elements = self.__parse_elements(
                xml=xml, image_path=screenshot_path, action=action_type
            )

            # 3. Generate Annotated Image
            annotated_path = path_manager.get_annotated_path(
                package_name=package_name, session_id=session_id, filename=f"{filename_base}.png"
            )
            annotated_result = self.__annotate(
                source=screenshot_path, destination=annotated_path, elements=elements
            )

            if not annotated_result:
                return screen, self.__label_map.copy()

            if self.__storage:
                with annotated_result.open("rb") as handle:
                    annotated_data = handle.read()

                self.__fire_and_forget(
                    self.__storage.save(
                        data=annotated_data,
                        metadata={
                            "category": "annotated",
                            "session_id": session_id,
                            "package_name": package_name,
                            "filename": f"{filename_base}.png",
                        },
                    )
                )

            # 4. Build Result Capture
            capture = self.__build_capture(original=screen, path=annotated_result)

            # Inject metadata
            new_metadata = capture.metadata.copy()

            new_metadata["xml_path"] = str(xml_path)
            new_metadata["path"] = str(annotated_result)
            capture = capture.model_copy(update={"metadata": new_metadata})

            duration = (datetime.now() - start_time).total_seconds()
            logger.info(
                f"Hierarchy processing complete in {duration:.2f}s. Elements found: {len(self.__label_map)}"
            )

            return capture, self.__label_map.copy()

        except Exception as exception:
            logger.exception(f"Hierarchy processing failed: {exception}")
            return screen, {}
        finally:
            if created_temporary_screenshot and screenshot_path is not None:
                with contextlib.suppress(FileNotFoundError):
                    screenshot_path.unlink()

    def __save_file(self, path: Path, data: bytes, mode: str = "wb") -> None:
        """
        Helper to save file.
        """

        with path.open(mode) as handle:
            handle.write(data)

    def __create_working_screenshot(self, *, image: bytes) -> Path:
        """
        Persist a temporary screenshot file for XML parsing and annotation only.
        """

        with tempfile.NamedTemporaryFile(mode="wb", suffix=".png", delete=False) as handle:
            handle.write(image)
            return Path(handle.name)

    def __resolve_source_screenshot_path(self, *, screen: ScreenCapture) -> Tuple[Path, bool]:
        """
        Resolve the screenshot path to use for hierarchy parsing and annotation.
        """

        raw_path = screen.metadata.get("path")
        if isinstance(raw_path, str):
            candidate = Path(raw_path)
            if candidate.exists():
                return candidate, False

        return self.__create_working_screenshot(image=screen.image), True

    def __parse_elements(
        self, xml: str, image_path: Path, action: Optional[ActionType]
    ) -> List[LabeledElement]:
        """
        Identifies elements from XML.
        """

        start = xml.find("<")
        end = xml.rfind(">")

        if start != -1 and end != -1:
            xml = xml[start : end + 1]

        root = ET.fromstring(xml)  # nosec
        elements, self.__label_map = BoundsGenerator.create_element(
            root=root, image_path=str(image_path), action=action or ActionType.TAP
        )
        return elements

    def __annotate(
        self, source: Path, destination: Path, elements: List[LabeledElement]
    ) -> Optional[Path]:
        """
        Annotates image with identified elements.
        """

        path = ImageAnnotator.annotate(
            image_path=str(source), output_path=str(destination), elements=elements
        )
        return Path(path) if path else None

    def __build_capture(self, original: ScreenCapture, path: Path) -> ScreenCapture:
        """
        Builds capture object from annotated image.
        """

        with path.open("rb") as handle:
            return ScreenCapture(
                image=handle.read(),
                width=original.width,
                height=original.height,
                activity=original.activity,
                timestamp=original.timestamp,
                metadata={"path": str(path)},
            )
