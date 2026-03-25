from __future__ import annotations

import hashlib
import io
import time
from logging import getLogger
from pathlib import Path
from typing import List, Optional

try:
    import cv2
    import numpy

    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    cv2 = None
    numpy = None

from PIL import Image

from fathom.constants.execution import VISUAL_HASH_LENGTH
from fathom.constants.screen import INTERACTION_TEXT_PREVIEW_LENGTH, ZERO_HASH
from fathom.interfaces.perception import PerceptionPort
from fathom.interfaces.storage import StoragePort
from fathom.processing.parsers.signature import HierarchySignatureBuilder
from fathom.schemas.screens import ScreenCapture
from fathom.schemas.ui import LabeledElement

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
    ) -> None:
        self.__storage = storage
        self.__perception = perception
        self.__hierarchy_signature_builder = hierarchy_signature_builder

        self.__session_id = session_id

    async def perceive(self, *, session_id: Optional[str] = None) -> ScreenCapture:
        """
        Capture current screen state via DevicePort.

        Returns:
            ScreenCapture with screenshot data
        """

        effective_session_id = session_id or self.__session_id

        if not effective_session_id:
            raise ValueError("session_id must be provided either in __init__ or perceive()")

        capture = await self.__perception.capture()

        # Store screenshot artifact with metadata for structured storage
        storage_id = await self.__persist_capture(
            data=capture.image,
            package_name=capture.activity,
            activity_name=capture.activity,
            session_id=effective_session_id,
        )
        metadata = dict(capture.metadata)
        metadata["storage_id"] = storage_id

        if self.__is_local_artifact_path(storage_id=storage_id):
            metadata["path"] = storage_id

        return capture.model_copy(update={"metadata": metadata})

    async def __persist_capture(
        self, *, data: bytes, session_id: str, package_name: str, activity_name: str
    ) -> str:
        """
        Persists screenshot to storage.
        """

        return await self.__storage.save(
            data=data,
            metadata={
                "type": "screenshot",
                "timestamp": time.time(),
                "session_id": session_id,
                "package_name": package_name,
                "activity_name": activity_name,
            },
        )

    def __is_local_artifact_path(self, *, storage_id: str) -> bool:
        """
        Determine whether the storage identifier points to a local filesystem artifact.
        """

        return Path(storage_id).is_absolute() and Path(storage_id).exists()

    def compute_visual_hash(self, *, capture: ScreenCapture) -> str:
        """
        Compute a robust Perceptual Hash (pHash) for the screen capture.
        Resilient to minor noise, status bar changes, and compression artifacts.
        Produces a 64-bit hex string compatible with Hamming distance.
        """

        try:
            if OPENCV_AVAILABLE:
                return self.__compute_phash_opencv(image_data=capture.image)

            return self.__compute_phash_pillow(image_data=capture.image)
        except Exception as exception:
            logger.warning(f"Could not compute pHash, falling back to SHA256: {exception}")
            return hashlib.sha256(capture.image).hexdigest()[:VISUAL_HASH_LENGTH]

    def __compute_phash_opencv(self, *, image_data: bytes) -> str:
        """
        Computes pHash using OpenCV and NumPy (Primary).
        """

        if not OPENCV_AVAILABLE or cv2 is None or numpy is None:
            raise RuntimeError("OpenCV not available")

        logger.debug("Computing pHash using OpenCV")

        image_array = numpy.frombuffer(image_data, numpy.uint8)
        decoded_image = cv2.imdecode(image_array, cv2.IMREAD_GRAYSCALE)

        if decoded_image is None:
            raise ValueError("Could not decode image with OpenCV")

        resized_image = cv2.resize(decoded_image, (32, 32), interpolation=cv2.INTER_AREA)
        float_image = numpy.float32(resized_image)
        dct_transform = cv2.dct(float_image)

        low_frequencies = dct_transform[0:8, 0:8]
        average_frequency = (numpy.sum(low_frequencies) - low_frequencies[0, 0]) / 63.0

        hash_integer = 0
        flattened_frequencies = (low_frequencies > average_frequency).flatten()

        for index, value in enumerate(flattened_frequencies):
            if value:
                hash_integer |= 1 << (63 - index)

        return f"{hash_integer:016x}"

    def __compute_phash_pillow(self, *, image_data: bytes) -> str:
        """
        Compute a lightweight fallback perceptual hash using Pillow.
        """

        logger.debug("Computing fallback perceptual hash using Pillow")

        with Image.open(io.BytesIO(image_data)) as pillow_image:
            grayscale_image = pillow_image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
            # getdata() returns values that need to be safely converted to int
            # Values can be int, float, tuple, or None - handle each case
            pixel_data: List[int] = []

            for pixel_value in grayscale_image.getdata():
                if isinstance(pixel_value, int):
                    pixel_data.append(pixel_value)

                elif isinstance(pixel_value, float):
                    pixel_data.append(int(pixel_value))

                elif isinstance(pixel_value, (tuple, list)) and len(pixel_value) > 0:
                    pixel_data.append(int(pixel_value[0]))

                else:
                    pixel_data.append(0)  # Fallback for None or unknown types

            hash_integer = 0

            for row_index in range(8):
                for col_index in range(8):
                    pixel_index = row_index * 9 + col_index
                    if pixel_data[pixel_index] > pixel_data[pixel_index + 1]:
                        hash_integer |= 1 << (63 - (row_index * 8 + col_index))

            return f"{hash_integer:016x}"

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
