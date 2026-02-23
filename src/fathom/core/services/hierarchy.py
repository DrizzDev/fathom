from __future__ import annotations

import xml.etree.ElementTree as ET  # nosec
from datetime import datetime
from logging import getLogger
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fathom.base.paths import SharedPathManager
from fathom.constants import ActionType
from fathom.interfaces.device import DevicePort
from fathom.processing.annotator import ImageAnnotator
from fathom.processing.drawer import BoundsGenerator
from fathom.schemas.screens import ScreenCapture
from fathom.schemas.ui import LabeledElement

logger = getLogger(__name__)


class HierarchyService:
    """
    Service responsible for UI hierarchy analysis. Optimized for high-speed grounding.
    """

    def __init__(self, device: DevicePort) -> None:
        """
        Initialize hierarchy service with device port."""

        self.__device = device
        self.__label_map: Dict[str, Any] = {}

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

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename_base = f"{timestamp}"

            # 1. Save Raw Screenshot (using path manager)
            screenshot_path = path_manager.get_screenshot_path(
                package_name=package_name, session_id=session_id, filename=f"{filename_base}.png"
            )
            self.__save_file(path=screenshot_path, data=screen.image, mode="wb")

            # 2. Save Raw XML (using path manager)
            xml_path = path_manager.get_xml_path(
                package_name=package_name, session_id=session_id, filename=f"{filename_base}.xml"
            )
            self.__save_file(path=xml_path, data=xml.encode("utf-8"), mode="wb")

            # 3. Parse Elements
            elements = self.__parse_elements(
                xml=xml, image_path=screenshot_path, action=action_type
            )

            # 4. Generate Annotated Image
            annotated_path = path_manager.get_annotated_path(
                package_name=package_name, session_id=session_id, filename=f"{filename_base}.png"
            )
            annotated_result = self.__annotate(
                source=screenshot_path, destination=annotated_path, elements=elements
            )

            if not annotated_result:
                return screen, self.__label_map.copy()

            # 5. Build Result Capture
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

    def __save_file(self, path: Path, data: bytes, mode: str = "wb") -> None:
        """
        Helper to save file.
        """

        with path.open(mode) as handle:
            handle.write(data)

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
