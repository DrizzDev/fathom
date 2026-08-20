from __future__ import annotations

import asyncio
import time
import xml.etree.ElementTree as ET  # nosec
from datetime import datetime
from logging import getLogger
from typing import TYPE_CHECKING, List, Optional

from fathom.base.paths import SharedPathManager
from fathom.constants import ActionType

if TYPE_CHECKING:
    from pathlib import Path
from fathom.core.artifact.pipeline import ArtifactPipeline
from fathom.interfaces.storage import StoragePort
from fathom.processing.annotator import ImageAnnotator
from fathom.processing.drawer import BoundsGenerator
from fathom.schemas.artifact import (
    AnnotatedPayload,
    ArtifactRecord,
    HierarchyXmlPayload,
)
from fathom.schemas.hierarchy import (
    HierarchyElementExtraction,
    HierarchyProcessingResult,
)
from fathom.schemas.perception import PerceptionConfiguration
from fathom.schemas.screens import ScreenCapture
from fathom.schemas.ui import LabeledElement

logger = getLogger(__name__)


class HierarchyService:
    """
    Parses UI-hierarchy XML into labeled interactive elements and annotated captures for grounding.
    """

    def __init__(
        self,
        storage: Optional[StoragePort] = None,
        *,
        pipeline: Optional[ArtifactPipeline] = None,
        configuration: Optional[PerceptionConfiguration] = None,
    ) -> None:
        """
        Initialize hierarchy service.

        ``configuration.cv.enabled`` toggles whether the OpenCV
        :class:`VisualControlLabeler` appends fallback boxes onto the XML
        manifest. Off by default so the original XML+LLM flow boots
        without an extra detector pass.
        """

        self.__storage = storage
        self.__pipeline = pipeline
        self.__configuration = configuration or PerceptionConfiguration()

    def extract_elements(
        self,
        *,
        xml: str,
        screen: ScreenCapture,
        action_type: Optional[ActionType] = None,
    ) -> List[LabeledElement]:
        """
        Extract interactive elements from XML without producing annotated artifacts.
        """

        extraction = self.__parse_elements(
            xml=xml,
            action=action_type,
            image=screen.image,
        )
        return extraction.labeled_elements

    async def process_xml_and_screen(
        self,
        xml: str,
        screen: ScreenCapture,
        *,
        session_id: str,
        package_name: str,
        step_number: int,
        path_manager: SharedPathManager,
        action_type: Optional[ActionType] = None,
    ) -> HierarchyProcessingResult:
        """
        Processes XML and screen data to identify UI elements.
        """

        start_time = datetime.now()
        xml_size_kb = len(xml.encode("utf-8")) / 1024

        logger.info(f"Processing hierarchy. XML Size: {xml_size_kb:.2f} KB, Action: {action_type}")

        if xml_size_kb < 0.2:
            logger.warning(f"XML too small ({xml_size_kb:.2f} KB), possibly invalid.")
            return HierarchyProcessingResult(
                label_map={},
                labeled_elements=[],
                annotated_capture=screen,
            )

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename_base = f"{timestamp}"

            xml_path = path_manager.get_xml_path(
                session_id=session_id, filename=f"{filename_base}__{package_name}.xml"
            )
            xml_bytes = xml.encode("utf-8")
            self.__save_file(path=xml_path, data=xml_bytes, mode="wb")

            await self.__emit_xml_artifact(
                content=xml,
                session_id=session_id,
                step_number=step_number,
                package_name=package_name,
            )

            extraction = await asyncio.to_thread(
                self.__parse_elements,
                xml=xml,
                action=action_type,
                image=screen.image,
            )
            labeled_elements = extraction.labeled_elements

            annotated_image = await asyncio.to_thread(
                ImageAnnotator.annotate,
                image=screen.image,
                elements=labeled_elements,
            )

            if not annotated_image:
                return HierarchyProcessingResult(
                    annotated_capture=screen,
                    label_map=extraction.label_map,
                    labeled_elements=labeled_elements,
                )

            capture = self.__build_capture(
                original=screen, annotated_image=annotated_image, xml_path=xml_path
            )
            if (
                staged_path := await self.__emit_annotated_artifact(
                    capture=capture,
                    session_id=session_id,
                    step_number=step_number,
                    package_name=package_name,
                )
            ) is not None:
                capture = capture.model_copy(update={"annotated_uri": str(staged_path)})

            duration = (datetime.now() - start_time).total_seconds()
            label_map_count = sum(
                1 for key in extraction.label_map if not str(key).startswith("__")
            )
            logger.info(
                "Hierarchy stage count",
                extra={
                    "stage": "summary",
                    "duration": round(duration, 3),
                    "count": len(labeled_elements),
                    "event": "hierarchy.stage.count",
                    "label_map_size": label_map_count,
                    "component": "core.services.hierarchy",
                },
            )

            return HierarchyProcessingResult(
                annotated_capture=capture,
                label_map=extraction.label_map,
                labeled_elements=labeled_elements,
            )

        except Exception as exception:
            logger.exception(f"Hierarchy processing failed: {exception}")
            return HierarchyProcessingResult(
                label_map={},
                labeled_elements=[],
                annotated_capture=screen,
            )

    def __save_file(self, *, path: Path, data: bytes, mode: str = "wb") -> None:
        """
        Write bytes to the given path using the requested file mode.
        """

        with path.open(mode) as handle:
            handle.write(data)

    async def __emit_xml_artifact(
        self,
        *,
        content: str,
        session_id: str,
        step_number: int,
        package_name: str,
    ) -> None:
        """
        Hand the hierarchy XML dump to the artifact pipeline for durable upload.
        """

        if self.__pipeline is None:
            return

        await self.__pipeline.emit(
            record=ArtifactRecord(
                session_id=session_id,
                package_name=package_name,
                step_number=step_number,
                created=int(time.time() * 1000),
                payload=HierarchyXmlPayload(content=content),
            ),
        )

    async def __emit_annotated_artifact(
        self,
        *,
        session_id: str,
        step_number: int,
        package_name: str,
        capture: ScreenCapture,
    ) -> Optional[Path]:
        """
        Hand the XML-annotated capture to the artifact pipeline and surface
        the staged path so the caller can stamp ``annotated_uri`` on the
        returned capture.
        """

        if self.__pipeline is None:
            return None

        return await self.__pipeline.emit(
            record=ArtifactRecord(
                session_id=session_id,
                step_number=step_number,
                package_name=package_name,
                created=int(time.time() * 1000),
                payload=AnnotatedPayload(capture=capture),
            ),
        )

    def __parse_elements(
        self,
        *,
        xml: str,
        image: bytes,
        action: Optional[ActionType],
    ) -> HierarchyElementExtraction:
        """
        Identify elements from XML using in-memory screenshot bytes.
        """

        start = xml.find("<")
        end = xml.rfind(">")

        if start != -1 and end != -1:
            xml = xml[start : end + 1]

        root = ET.fromstring(xml)  # nosec
        elements, label_map = BoundsGenerator.create_element(
            root=root,
            image=image,
            action=action or ActionType.TAP,
            cv_enabled=self.__configuration.cv.enabled,
        )
        return HierarchyElementExtraction(label_map=label_map, labeled_elements=elements)

    def __build_capture(
        self,
        *,
        xml_path: Path,
        annotated_image: bytes,
        original: ScreenCapture,
    ) -> ScreenCapture:
        """
        Build a capture carrying the annotated bytes alongside the original screen.
        """

        metadata = dict(original.metadata)
        metadata["xml_path"] = str(xml_path)

        return ScreenCapture(
            metadata=metadata,
            state=original.state,
            image=original.image,
            width=original.width,
            height=original.height,
            activity=original.activity,
            timestamp=original.timestamp,
            annotated_image=annotated_image,
            xml_content=original.xml_content,
            annotated_uri=original.annotated_uri,
            screenshot_uri=original.screenshot_uri,
        )
