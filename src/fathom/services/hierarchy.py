from __future__ import annotations

import time
import xml.etree.ElementTree as ET  # nosec
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from fathom.constants import ActionType
from fathom.schemas.screens import ScreenCapture
from fathom.tools.device import DeviceTool
from fathom.tools.vision.processing.annotator import ImageAnnotator
from fathom.tools.vision.processing.drawer import BoundingBoxGenerator


class HierarchyService:
    """
    Service responsible for UI hierarchy analysis and screenshot annotation.
    Follows single responsibility principle for UI grounding.
    """

    def __init__(self, device: DeviceTool) -> None:
        self.__device = device
        self.__label_map: Dict[str, Any] = {}

    @property
    def label_map(self) -> Dict[str, Any]:
        """
        Returns a copy of the current label mapping.
        """
        return self.__label_map.copy()

    async def process_screen(
        self, screen: ScreenCapture, action_type: Optional[ActionType] = None
    ) -> Tuple[Optional[ScreenCapture], float, float]:
        """
        Dumps hierarchy, generates elements, and returns an annotated ScreenCapture.
        Returns: (Annotated Capture, Dump Duration, Processing Duration)
        """

        start_timestamp = time.time()
        xml_content = await self.__device.dump_hierarchy()
        dump_duration = time.time() - start_timestamp

        if not xml_content:
            return None, dump_duration, 0.0

        processing_start_timestamp = time.time()
        try:
            timestamp = int(time.time() * 1000)
            temporary_path = Path(f"assets/screenshot/temp_{timestamp}.png")
            temporary_path.parent.mkdir(parents=True, exist_ok=True)

            with temporary_path.open("wb") as file_handle:
                file_handle.write(screen.image)

            # Sanitize XML
            xml_start_index = xml_content.find("<")
            xml_end_index = xml_content.rfind(">")
            if xml_start_index != -1 and xml_end_index != -1:
                xml_content = xml_content[xml_start_index : xml_end_index + 1]

            root_element = ET.fromstring(xml_content)  # nosec
            elements, self.__label_map = BoundingBoxGenerator.create_element(
                root_element, str(temporary_path), action=action_type or ActionType.TAP
            )

            annotated_filename = f"assets/annotated/xml_annotated_{timestamp}.png"
            annotated_path = ImageAnnotator.annotate(
                str(temporary_path), annotated_filename, elements
            )

            if not annotated_path:
                return None, dump_duration, time.time() - processing_start_timestamp

            with Path(annotated_path).open("rb") as file_handle:
                annotated_capture = ScreenCapture(
                    width=screen.width,
                    height=screen.height,
                    image=file_handle.read(),
                    activity=screen.activity,
                    timestamp=screen.timestamp,
                )

            return (
                annotated_capture,
                dump_duration,
                time.time() - processing_start_timestamp,
            )

        except Exception:  # nosec
            return None, dump_duration, time.time() - processing_start_timestamp