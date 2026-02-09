from __future__ import annotations

import xml.etree.ElementTree as ET  # nosec
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fathom.constants import ActionType
from fathom.schemas.screens import ScreenCapture
from fathom.schemas.ui import LabeledElement
from fathom.tools.device import DeviceTool
from fathom.tools.vision.processing.annotator import ImageAnnotator
from fathom.tools.vision.processing.drawer import BoundingBoxGenerator


class HierarchyService:
    """
    Service responsible for UI hierarchy analysis.
    Optimized for high-speed grounding.
    """

    def __init__(self, device: DeviceTool) -> None:
        self.__device = device
        self.__label_map: Dict[str, Any] = {}

    @property
    def label_map(self) -> Dict[str, Any]:
        """
        Returns label mapping.
        """
        return self.__label_map.copy()

    async def process_xml_and_screen(
        self, screen: ScreenCapture, xml: str, action_type: Optional[ActionType] = None
    ) -> Tuple[Optional[ScreenCapture], Dict[str, Any]]:
        """
        Processes existing XML and screen data.
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            screenshot_path = self.__save_screenshot(screen.image, timestamp)

            # Persist raw XML in background
            self.__save_xml(xml, timestamp)

            elements = self.__parse_elements(xml, screenshot_path, action_type)
            annotated_path = self.__annotate(screenshot_path, timestamp, elements)

            if not annotated_path:
                return None, {}

            capture = self.__build_capture(original=screen, path=annotated_path)

            # Inject path into metadata
            new_metadata = capture.metadata.copy()
            new_metadata["path"] = str(annotated_path)
            capture = capture.model_copy(update={"metadata": new_metadata})

            return capture, self.__label_map.copy()

        except Exception:  # nosec
            return None, {}

    def __save_xml(self, content: str, timestamp: str) -> Path:
        """
        Persists hierarchy.
        """
        directory = Path("assets/xmls")
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{timestamp}.xml"
        with path.open("w", encoding="utf-8") as handle:
            handle.write(content)
        return path

    def __save_screenshot(self, data: bytes, timestamp: str) -> Path:
        """
        Persists screenshot.
        """
        directory = Path("assets/screenshot")
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{timestamp}.png"
        with path.open("wb") as handle:
            handle.write(data)
        return path

    def __parse_elements(
        self, xml: str, image_path: Path, action: Optional[ActionType]
    ) -> List[LabeledElement]:
        """
        Identifies elements.
        """
        start = xml.find("<")
        end = xml.rfind(">")
        if start != -1 and end != -1:
            xml = xml[start : end + 1]

        root = ET.fromstring(xml)  # nosec
        elements, self.__label_map = BoundingBoxGenerator.create_element(
            root, str(image_path), action=action or ActionType.TAP
        )
        return elements

    def __annotate(
        self, source: Path, timestamp: str, elements: List[LabeledElement]
    ) -> Optional[Path]:
        """
        Annotates image.
        """
        directory = Path("assets/annotated")
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / f"{timestamp}.png"
        path = ImageAnnotator.annotate(str(source), str(destination), elements)
        return Path(path) if path else None

    def __build_capture(self, original: ScreenCapture, path: Path) -> ScreenCapture:
        """
        Builds capture object.
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
