from __future__ import annotations

import xml.etree.ElementTree as ET  # nosec
from datetime import datetime
from logging import getLogger
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fathom.constants import ActionType
from fathom.schemas.screens import ScreenCapture
from fathom.schemas.ui import LabeledElement
from fathom.tools.device import DeviceTool
from fathom.tools.vision.processing.annotator import ImageAnnotator
from fathom.tools.vision.processing.drawer import BoundsGenerator

logger = getLogger(__name__)


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

        start_time = datetime.now()
        xml_size_kb = len(xml.encode("utf-8")) / 1024

        logger.info(f"Processing hierarchy. XML Size: {xml_size_kb:.2f} KB, Action: {action_type}")

        if xml_size_kb < 0.2:  # Very likely empty or error message
            logger.warning(
                f"XML too small ({xml_size_kb:.2f} KB), possibly invalid or loading state."
            )
            return screen, {}

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # 1. Save Raw Screenshot
            screenshot_path = self.__save_screenshot(data=screen.image, timestamp=timestamp)
            logger.debug(f"Saved raw screenshot to: {screenshot_path}")

            # 2. Save Raw XML
            xml_path = self.__save_xml(content=xml, timestamp=timestamp)
            logger.debug(f"Saved raw XML to: {xml_path}")

            # 3. Parse Elements
            elements = self.__parse_elements(
                xml=xml, image_path=screenshot_path, action=action_type
            )
            logger.info(f"Found {len(elements)} elements in hierarchy.")

            # 4. Generate Annotated Image
            annotated_path = self.__annotate(
                source=screenshot_path, timestamp=timestamp, elements=elements
            )

            if not annotated_path:
                logger.warning("Annotation failed, returning original screen.")
                return screen, self.__label_map.copy()

            logger.info(f"Saved annotated screenshot to: {annotated_path}")

            # 5. Build Result Capture
            capture = self.__build_capture(original=screen, path=annotated_path)

            # Inject metadata
            new_metadata = capture.metadata.copy()
            new_metadata["xml_path"] = str(xml_path)
            new_metadata["path"] = str(annotated_path)
            capture = capture.model_copy(update={"metadata": new_metadata})

            duration = (datetime.now() - start_time).total_seconds()
            logger.info(f"Hierarchy processing complete in {duration:.2f}s")

            return capture, self.__label_map.copy()

        except Exception as exception:
            logger.exception(f"Hierarchy processing failed: {exception}")
            return screen, {}

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
        elements, self.__label_map = BoundsGenerator.create_element(
            root=root, image_path=str(image_path), action=action or ActionType.TAP
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

        path = ImageAnnotator.annotate(
            image_path=str(source), output_path=str(destination), elements=elements
        )
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
