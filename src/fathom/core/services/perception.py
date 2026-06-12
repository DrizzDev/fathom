from __future__ import annotations

import asyncio  # noqa: TC003 — used at runtime for Task types
import hashlib
import time
from logging import getLogger
from typing import TYPE_CHECKING, Any, List, Optional

from fathom.constants.execution import VISUAL_HASH_LENGTH
from fathom.constants.screen import INTERACTION_TEXT_PREVIEW_LENGTH, ZERO_HASH
from fathom.core.artifact.pipeline import ArtifactPipeline
from fathom.core.exceptions import ConfigurationError
from fathom.core.perception.hashing import VisualHashEngine
from fathom.interfaces.perception import PerceptionPort
from fathom.interfaces.storage import StoragePort
from fathom.processing.parsers.signature import HierarchySignatureBuilder
from fathom.schemas.artifact import ArtifactRecord, ScreenshotPayload
from fathom.schemas.screens import ScreenCapture, ScreenState
from fathom.schemas.ui import LabeledElement

if TYPE_CHECKING:
    from pathlib import Path

logger = getLogger(__name__)


class PerceptionService:
    """
    Perception service for capturing and hashing screen state.
    """

    def __init__(
        self,
        storage: StoragePort,
        perception: PerceptionPort,
        hierarchy_signature_builder: HierarchySignatureBuilder,
        *,
        session_id: Optional[str] = None,
        pipeline: Optional[ArtifactPipeline] = None,
        visual_hash_engine: Optional[VisualHashEngine] = None,
    ) -> None:
        self.__storage = storage
        self.__pipeline = pipeline
        self.__perception = perception
        self.__session_id = session_id

        self.__hierarchy_signature_builder = hierarchy_signature_builder
        self.__visual_hash_engine = visual_hash_engine or VisualHashEngine()

        self.__background_tasks: set[asyncio.Task[Any]] = set()

    async def perceive(
        self,
        *,
        session_id: Optional[str] = None,
        step_number: int,
    ) -> ScreenCapture:
        """
        Capture the current screen, emit it through the artifact pipeline,
        and stamp the staged path onto ``capture.screenshot_uri`` as the
        canonical handle. Downstream services read the URI off the typed field; raw bytes ride on ``capture.image``.
        """

        if not (effective_session_id := session_id or self.__session_id):
            raise ConfigurationError("session_id must be provided either in __init__ or perceive()")

        capture = await self.__perception.capture()

        staged_path = await self.__emit_screenshot_artifact(
            capture=capture,
            session_id=effective_session_id,
            step_number=step_number,
        )
        if staged_path is None:
            return await self.__fallback_persist_capture(
                capture=capture,
                session_id=effective_session_id,
            )

        return capture.model_copy(update={"screenshot_uri": str(staged_path)})

    async def __emit_screenshot_artifact(
        self,
        *,
        capture: ScreenCapture,
        session_id: str,
        step_number: int,
    ) -> Optional[Path]:
        """
        Stage the screen capture through the artifact pipeline and return the EFS-staged path.
        Caller stamps the path onto ``capture.screenshot_uri`` so downstream consumers read bytes from local disk.
        """

        if self.__pipeline is None:
            return None

        staged_path = await self.__pipeline.emit(
            record=ArtifactRecord(
                session_id=session_id,
                step_number=step_number,
                package_name=capture.activity,
                created=int(time.time() * 1000),
                payload=ScreenshotPayload(capture=capture),
            ),
        )

        if staged_path is None:
            return None

        logger.info(
            "Pre-action screenshot artifact staged",
            extra={
                "component": "perception.service",
                "session.id": session_id,
                "step.number": step_number,
                "artifact.kind": "screenshot",
                "artifact.path": str(staged_path),
                "event": "perception.screenshot.staged",
            },
        )
        return staged_path

    async def __fallback_persist_capture(
        self,
        *,
        session_id: str,
        capture: ScreenCapture,
    ) -> ScreenCapture:
        """
        Persist via :class:`StoragePort` when no artifact pipeline is wired.

        Production runs always have the pipeline configured; this branch
        exists so unit tests and minimal embeddings that omit the
        pipeline still produce a usable storage identifier. The remote
        identifier returned by :class:`StoragePort` is a stable handle
        and is stamped onto ``screenshot_uri`` (a typed field) so
        downstream consumers do not need to peek into ``metadata``.
        """

        storage_id = await self.__storage.save(
            data=capture.image,
            metadata={
                "type": "screenshot",
                "phase": "pre_action",
                "timestamp": time.time(),
                "session_id": session_id,
                "package_name": capture.activity,
                "activity_name": capture.activity,
            },
        )
        metadata = dict(capture.metadata)
        metadata["storage_id"] = storage_id

        return capture.model_copy(
            update={
                "metadata": metadata,
                "screenshot_uri": storage_id,
            }
        )

    def compute_visual_hash(self, *, capture: ScreenCapture) -> str:
        """
        Compute a perceptual hash for the screen capture via the injected hash engine.
        """

        return self.__visual_hash_engine.hash(image=capture.image)

    def compute_xml_hash(self, *, capture: ScreenCapture) -> str:
        """
        Compute a structural hash from the normalized XML hierarchy.
        """

        xml_content = capture.xml_content

        if not xml_content:
            return ZERO_HASH

        try:
            return self.__hierarchy_signature_builder.compute_hash(xml_content=xml_content)
        except Exception as exception:
            logger.warning(f"Could not compute xml_hash: {exception}")
            return ZERO_HASH

    def compute_interaction_hash(
        self,
        *,
        elements: Optional[List[LabeledElement]] = None,
    ) -> str:
        """
        Compute a stable hash of interactive elements based strictly on their semantic identity (Class, ID, Text, Desc).
        """

        if not elements:
            return ZERO_HASH

        try:
            stable_identities = self.__extract_element_identities(elements=elements)

            if not stable_identities:
                return ZERO_HASH

            # Sort to ensure order independence
            stable_identities.sort()
            combined_signature = "\n".join(stable_identities)

            return hashlib.md5(
                combined_signature.encode("utf-8"), usedforsecurity=False
            ).hexdigest()[:VISUAL_HASH_LENGTH]

        except Exception as exception:
            logger.warning(f"Could not compute interaction_hash: {exception}")
            return ZERO_HASH

    def build_state(
        self,
        *,
        capture: ScreenCapture,
        elements: Optional[List[LabeledElement]] = None,
    ) -> ScreenState:
        """
        Composes a full MLSIA screen state (activity, visual, xml, interaction hashes).
        """

        activity_hash = hashlib.md5(capture.activity.encode(), usedforsecurity=False).hexdigest()[
            :VISUAL_HASH_LENGTH
        ]

        return ScreenState(
            activity=capture.activity,
            timestamp=capture.timestamp,
            activity_hash=activity_hash,
            visual_hash=self.compute_visual_hash(capture=capture),
            xml_hash=self.compute_xml_hash(capture=capture),
            interaction_hash=self.compute_interaction_hash(elements=elements),
        )

    def __extract_element_identities(self, elements: List[LabeledElement]) -> List[str]:
        """
        Extracts semantic identities from a collection of UI elements.
        """

        stable_identities: List[str] = []

        for element_info in elements:
            attributes = element_info.attributes
            if not attributes:
                continue

            raw_class = str(attributes.get("class", "")).strip()
            raw_type = str(attributes.get("type", "")).strip()

            if raw_class:
                class_name = raw_class.split(".")[-1]
                identifier = str(attributes.get("resource-id", "")).split("/")[-1]
                element_text = str(attributes.get("text", "")).strip()[
                    :INTERACTION_TEXT_PREVIEW_LENGTH
                ]
                element_description = str(attributes.get("content-desc", "")).strip()[
                    :INTERACTION_TEXT_PREVIEW_LENGTH
                ]
            else:
                class_name = raw_type.replace("XCUIElementType", "") or raw_type or "Unknown"
                identifier = str(attributes.get("name", "")).strip()[
                    :INTERACTION_TEXT_PREVIEW_LENGTH
                ]
                label = str(attributes.get("label", "")).strip()[:INTERACTION_TEXT_PREVIEW_LENGTH]
                value = str(attributes.get("value", "")).strip()[:INTERACTION_TEXT_PREVIEW_LENGTH]
                element_text = label if label else identifier
                element_description = value

            semantic_identity = f"{class_name}|{identifier}|{element_text}|{element_description}"
            stable_identities.append(semantic_identity)

        return stable_identities
