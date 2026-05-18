from __future__ import annotations

import asyncio
import contextlib
import tempfile
import time
import xml.etree.ElementTree as ET  # nosec
from datetime import datetime
from logging import getLogger
from pathlib import Path
from typing import Any, List, Optional, Set

from fathom.base.paths import SharedPathManager
from fathom.constants import DRAIN_TIMEOUT, ActionType
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
    ResolvedHierarchyScreenshot,
)
from fathom.schemas.screens import ScreenCapture
from fathom.schemas.ui import LabeledElement

logger = getLogger(__name__)


class HierarchyService:
    """
    Service responsible for UI hierarchy analysis. Optimized for high-speed grounding.
    """

    def __init__(
        self,
        storage: Optional[StoragePort] = None,
        *,
        pipeline: Optional[ArtifactPipeline] = None,
        cv_enabled: bool = False,
    ) -> None:
        """
        Initialize hierarchy service.

        ``cv_enabled`` toggles whether the OpenCV visual-control labeler
        appends fallback boxes onto the XML manifest. Off by default so
        the original XML+LLM flow boots without an extra detector pass.
        """

        self.__storage = storage
        self.__pipeline = pipeline
        self.__cv_enabled = cv_enabled
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

    async def drain_background_tasks(self) -> None:
        """
        Await all pending background upload tasks with a bounded timeout.
        """

        pending = [task for task in self.__background_tasks if not task.done()]
        if not pending:
            return

        logger.info(
            f"[HierarchyService] draining {len(pending)} background tasks (timeout={DRAIN_TIMEOUT}s)"
        )

        try:
            await asyncio.wait_for(
                asyncio.gather(*pending, return_exceptions=True),
                timeout=DRAIN_TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.warning(
                f"[HierarchyService] drain timed out, cancelling {len(pending)} remaining tasks"
            )
            for task in pending:
                if not task.done():
                    task.cancel()

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

        resolved_screenshot: Optional[ResolvedHierarchyScreenshot] = None

        try:
            resolved_screenshot = self.__resolve_source_screenshot_path(screen=screen)
            element_extraction = self.__parse_elements(
                xml=xml,
                action=action_type,
                image_path=Path(resolved_screenshot.path),
            )
            return element_extraction.labeled_elements
        finally:
            if resolved_screenshot is not None and resolved_screenshot.created_temporary_file:
                with contextlib.suppress(FileNotFoundError):
                    Path(resolved_screenshot.path).unlink()

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

        resolved_screenshot: Optional[ResolvedHierarchyScreenshot] = None

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename_base = f"{timestamp}"

            resolved_screenshot = self.__resolve_source_screenshot_path(screen=screen)

            # 1. Save Raw XML (using path manager)
            xml_path = path_manager.get_xml_path(
                package_name=package_name, session_id=session_id, filename=f"{filename_base}.xml"
            )
            xml_bytes = xml.encode("utf-8")
            self.__save_file(path=xml_path, data=xml_bytes, mode="wb")

            await self.__emit_xml_artifact(
                content=xml,
                session_id=session_id,
                package_name=package_name,
                step_number=step_number,
            )

            # 2. Parse Elements (threaded to avoid blocking event loop)
            element_extraction = await asyncio.to_thread(
                self.__parse_elements,
                xml=xml,
                action=action_type,
                image_path=Path(resolved_screenshot.path),
            )
            labeled_elements = element_extraction.labeled_elements

            # 3. Generate Annotated Image. The legacy ImageAnnotator
            # write path is bypassed via a temp output file — only the
            # in-memory bytes survive, and the artifact pipeline owns
            # the durable copy under its filename grammar.
            annotated_result = await self.__annotate_to_temp(
                elements=labeled_elements,
                source=Path(resolved_screenshot.path),
            )

            if not annotated_result:
                return HierarchyProcessingResult(
                    annotated_capture=screen,
                    labeled_elements=labeled_elements,
                    label_map=element_extraction.label_map,
                )

            # 4. Build Result Capture
            capture = self.__build_capture(original=screen, path=annotated_result)
            await self.__emit_annotated_artifact(
                capture=capture,
                session_id=session_id,
                package_name=package_name,
                step_number=step_number,
            )

            # Drop the temp file once the bytes are on the capture and
            # the pipeline has them queued — no on-disk legacy copy.
            with contextlib.suppress(FileNotFoundError, OSError):
                annotated_result.unlink()

            # Inject metadata
            new_metadata = capture.metadata.copy()

            new_metadata["xml_path"] = str(xml_path)
            capture = capture.model_copy(update={"metadata": new_metadata})

            duration = (datetime.now() - start_time).total_seconds()
            # Align this terminal summary with the per-stage logs emitted
            # inside :class:`BoundsGenerator`: same ``hierarchy.stage.count``
            # event name, same field schema. Downstream log queries that
            # filter on ``event="hierarchy.stage.count"`` will pick up
            # every stage including the final ``summary`` row without
            # needing a second predicate.
            label_map_count = sum(
                1 for key in element_extraction.label_map if not str(key).startswith("__")
            )
            logger.info(
                "Hierarchy stage count",
                extra={
                    "component": "core.services.hierarchy",
                    "event": "hierarchy.stage.count",
                    "stage": "summary",
                    "count": len(labeled_elements),
                    "label_map_size": label_map_count,
                    "duration_s": round(duration, 3),
                },
            )

            return HierarchyProcessingResult(
                annotated_capture=capture,
                labeled_elements=labeled_elements,
                label_map=element_extraction.label_map,
            )

        except Exception as exception:
            logger.exception(f"Hierarchy processing failed: {exception}")
            return HierarchyProcessingResult(
                label_map={},
                labeled_elements=[],
                annotated_capture=screen,
            )
        finally:
            if resolved_screenshot is not None and resolved_screenshot.created_temporary_file:
                with contextlib.suppress(FileNotFoundError):
                    Path(resolved_screenshot.path).unlink()

    def __save_file(self, *, path: Path, data: bytes, mode: str = "wb") -> None:
        """
        Helper to save file.
        """

        with path.open(mode) as handle:
            handle.write(data)

    async def __emit_xml_artifact(
        self,
        *,
        content: str,
        session_id: str,
        package_name: str,
        step_number: int,
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
        capture: ScreenCapture,
        session_id: str,
        package_name: str,
        step_number: int,
    ) -> None:
        """
        Hand the XML-annotated capture to the artifact pipeline for durable upload.
        """

        if self.__pipeline is None:
            return

        await self.__pipeline.emit(
            record=ArtifactRecord(
                session_id=session_id,
                package_name=package_name,
                step_number=step_number,
                created=int(time.time() * 1000),
                payload=AnnotatedPayload(capture=capture),
            ),
        )

    def __create_working_screenshot(self, *, image: bytes) -> Path:
        """
        Persist a temporary screenshot file for XML parsing and annotation only.
        """

        with tempfile.NamedTemporaryFile(mode="wb", suffix=".png", delete=False) as handle:
            handle.write(image)
            return Path(handle.name)

    def __resolve_source_screenshot_path(
        self, *, screen: ScreenCapture
    ) -> ResolvedHierarchyScreenshot:
        """
        Resolve the screenshot path to use for hierarchy parsing and annotation.
        """

        raw_path = screen.metadata.get("raw_path")

        if isinstance(raw_path, str):
            candidate = Path(raw_path)
            if candidate.exists():
                return ResolvedHierarchyScreenshot(
                    path=str(candidate),
                    created_temporary_file=False,
                )

        if screen.annotated_image is None:
            rendered_path = screen.metadata.get("path")
            if isinstance(rendered_path, str):
                candidate = Path(rendered_path)
                if candidate.exists():
                    return ResolvedHierarchyScreenshot(
                        path=str(candidate),
                        created_temporary_file=False,
                    )

        return ResolvedHierarchyScreenshot(
            created_temporary_file=True,
            path=str(self.__create_working_screenshot(image=screen.image)),
        )

    def __parse_elements(
        self,
        *,
        xml: str,
        image_path: Path,
        action: Optional[ActionType],
    ) -> HierarchyElementExtraction:
        """
        Identifies elements from XML.
        Offloaded to thread pool to avoid blocking the event loop.
        """

        start = xml.find("<")
        end = xml.rfind(">")

        if start != -1 and end != -1:
            xml = xml[start : end + 1]

        root = ET.fromstring(xml)  # nosec
        elements, label_map = BoundsGenerator.create_element(
            root=root,
            image_path=str(image_path),
            action=action or ActionType.TAP,
            cv_enabled=self.__cv_enabled,
        )
        return HierarchyElementExtraction(label_map=label_map, labeled_elements=elements)

    async def __annotate(
        self,
        *,
        source: Path,
        destination: Path,
        elements: List[LabeledElement],
    ) -> Optional[Path]:
        """
        Annotates image with identified elements.
        Offloaded to thread pool to avoid blocking the event loop.
        """

        path = await asyncio.to_thread(
            ImageAnnotator.annotate,
            image_path=str(source),
            output_path=str(destination),
            elements=elements,
        )
        return Path(path) if path else None

    async def __annotate_to_temp(
        self,
        *,
        source: Path,
        elements: List[LabeledElement],
    ) -> Optional[Path]:
        """
        Render the manifest annotation to a temporary file.

        The bytes survive (read into ``capture.annotated_image`` and
        emitted via the pipeline). The on-disk legacy copy is deleted
        immediately so the durable artifact lives only under the
        pipeline's filename grammar.
        """

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
            destination = Path(handle.name)
        return await self.__annotate(source=source, destination=destination, elements=elements)

    def __build_capture(
        self,
        *,
        path: Path,
        original: ScreenCapture,
    ) -> ScreenCapture:
        """
        Builds capture object from annotated image.
        """

        with path.open("rb") as handle:
            metadata = dict(original.metadata)
            raw_path = original.metadata.get("raw_path") or original.metadata.get("path")

            if isinstance(raw_path, str):
                metadata["raw_path"] = raw_path

            metadata["path"] = str(path)

            return ScreenCapture(
                metadata=metadata,
                state=original.state,
                image=original.image,
                width=original.width,
                height=original.height,
                activity=original.activity,
                timestamp=original.timestamp,
                annotated_image=handle.read(),
                xml_content=original.xml_content,
            )
