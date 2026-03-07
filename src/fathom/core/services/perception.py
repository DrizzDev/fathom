from __future__ import annotations

import hashlib
import io
import time
import xml.etree.ElementTree as ElementTree  # nosec
from logging import getLogger
from typing import Any, Dict, Optional

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
from fathom.interfaces.device import DevicePort
from fathom.interfaces.storage import StoragePort
from fathom.interfaces.telemetry import TelemetryPort
from fathom.schemas.screens import ScreenCapture

logger = getLogger(__name__)


class PerceptionService:
    """
    Perception service for capturing and hashing screen state.
    """

    def __init__(
        self,
        device: DevicePort,
        storage: StoragePort,
        telemetry: TelemetryPort,
        session_id: Optional[str] = None,
    ) -> None:
        self.__device = device
        self.__storage = storage
        self.__telemetry = telemetry
        self.__session_id = session_id

    async def perceive(self, session_id: Optional[str] = None) -> ScreenCapture:
        """
        Capture current screen state via DevicePort.

        Returns:
            ScreenCapture with screenshot data
        """

        effective_session_id = session_id or self.__session_id
        if not effective_session_id:
            raise ValueError("session_id must be provided either in __init__ or perceive()")

        screenshot_bytes = await self.__device.capture_screen()
        width, height = await self.__device.get_dimensions()

        # Get current activity
        try:
            activity = await self.__device.get_current_package()
        except Exception as exception:
            await self.__telemetry.warning("Failed to get current package", error=str(exception))
            activity = "unknown"

        # Store screenshot artifact with metadata for structured storage
        storage_id = await self.__persist_capture(
            data=screenshot_bytes,
            package_name=activity,
            activity_name=activity,
            session_id=effective_session_id,
        )

        return ScreenCapture(
            width=width,
            height=height,
            activity=activity,
            image=screenshot_bytes,
            timestamp=int(time.time() * 1000),
            metadata={"storage_id": storage_id},
        )

    async def __persist_capture(
        self, data: bytes, package_name: str, activity_name: str, session_id: str
    ) -> str:
        """
        Persists screenshot to storage.
        """

        return await self.__storage.save(
            data=data,
            metadata={
                "type": "screenshot",
                "timestamp": time.time(),
                "package_name": package_name,
                "activity_name": activity_name,
                "session_id": session_id,
            },
        )

    def compute_visual_hash(self, capture: ScreenCapture) -> str:
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

    def __compute_phash_opencv(self, image_data: bytes) -> str:
        """
        Computes pHash using OpenCV and NumPy (Primary).
        """

        if not OPENCV_AVAILABLE or cv2 is None or numpy is None:
            raise RuntimeError("OpenCV not available")

        logger.info("Computing pHash using OpenCV")

        image_array: Any = numpy.frombuffer(image_data, numpy.uint8)
        decoded_image: Any = cv2.imdecode(image_array, cv2.IMREAD_GRAYSCALE)

        if decoded_image is None:
            raise ValueError("Could not decode image with OpenCV")

        resized_image: Any = cv2.resize(decoded_image, (32, 32), interpolation=cv2.INTER_AREA)
        float_image: Any = numpy.float32(resized_image)
        dct_transform: Any = cv2.dct(float_image)

        low_frequencies: Any = dct_transform[0:8, 0:8]
        average_frequency: Any = (numpy.sum(low_frequencies) - low_frequencies[0, 0]) / 63.0

        hash_integer = 0
        flattened_frequencies = (low_frequencies > average_frequency).flatten()

        for index, value in enumerate(flattened_frequencies):
            if value:
                hash_integer |= 1 << (63 - index)

        return f"{hash_integer:016x}"

    def __compute_phash_pillow(self, image_data: bytes) -> str:
        """
        Computes dHash using PIL (Fallback).
        """

        logger.info("Computing pHash using Pillow")

        with Image.open(io.BytesIO(image_data)) as pillow_image:
            grayscale_image = pillow_image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
            # getdata() returns values that need to be safely converted to int
            # Values can be int, float, tuple, or None - handle each case
            pixel_data: list[int] = []
            for p in grayscale_image.getdata():
                if isinstance(p, int):
                    pixel_data.append(p)
                elif isinstance(p, float):
                    pixel_data.append(int(p))
                elif isinstance(p, (tuple, list)) and len(p) > 0:
                    pixel_data.append(int(p[0]))
                else:
                    pixel_data.append(0)  # Fallback for None or unknown types

            hash_integer = 0
            for row_index in range(8):
                for col_index in range(8):
                    pixel_index = row_index * 9 + col_index
                    if pixel_data[pixel_index] > pixel_data[pixel_index + 1]:
                        hash_integer |= 1 << (63 - (row_index * 8 + col_index))

            return f"{hash_integer:016x}"

    def __get_tree_signature(self, node: ElementTree.Element, depth: int = 0) -> str:
        """
        Recursive helper to generate a structural signature for an XML node.
        Ignores dynamic system elements.
        """

        resource_id = str(node.get("resource-id", "")).split("/")[-1]

        if any(
            ignored_element in resource_id.lower()
            for ignored_element in ["systemui", "statusbar", "navigationbar"]
        ):
            return ""

        class_name = str(node.get("class", node.tag)).split(".")[-1]

        # Content-awareness: Include text and description to catch state changes
        # Truncation prevents signature explosion while maintaining uniqueness
        element_text = str(node.get("text", "")).strip()[:32]
        element_desc = str(node.get("content-desc", "")).strip()[:32]

        signature = f"{depth}:{class_name}#{resource_id}[{element_text}|{element_desc}]"

        child_signatures = []

        for child_node in node:
            if child_signature := self.__get_tree_signature(child_node, depth + 1):
                child_signatures.append(child_signature)

        if child_signatures:
            return f"({signature}[" + ",".join(child_signatures) + "])"

        return signature

    def compute_xml_hash(self, capture: ScreenCapture) -> str:
        """
        Compute a robust structural hash of the XML skeleton.
        Builds a tree signature using only Class and Resource-ID.
        """

        xml_content = capture.xml_content

        if not xml_content:
            return "0" * VISUAL_HASH_LENGTH

        try:
            start_index = xml_content.find("<")
            end_index = xml_content.rfind(">")

            if start_index != -1 and end_index != -1:
                xml_content = xml_content[start_index : end_index + 1]

            root_node = ElementTree.fromstring(xml_content)  # nosec
            tree_signature = self.__get_tree_signature(root_node)

            return hashlib.md5(tree_signature.encode("utf-8"), usedforsecurity=False).hexdigest()[
                :VISUAL_HASH_LENGTH
            ]

        except Exception as exception:
            logger.warning(f"Could not compute xml_hash: {exception}")
            return "0" * VISUAL_HASH_LENGTH

    def compute_interaction_hash(self, elements: Optional[Dict[str, Any]] = None) -> str:
        """
        Compute a stable hash of interactive elements based strictly on their semantic identity (Class, ID, Text, Desc).
        """

        if not elements:
            return "0" * VISUAL_HASH_LENGTH

        try:
            stable_identities = self.__extract_element_identities(elements=elements)

            if not stable_identities:
                return "0" * VISUAL_HASH_LENGTH

            # Sort to ensure order independence
            stable_identities.sort()
            combined_signature = "\n".join(stable_identities)

            return hashlib.md5(
                combined_signature.encode("utf-8"), usedforsecurity=False
            ).hexdigest()[:VISUAL_HASH_LENGTH]

        except Exception as exception:
            logger.warning(f"Could not compute interaction_hash: {exception}")
            return "0" * VISUAL_HASH_LENGTH

    def __extract_element_identities(self, elements: Dict[str, Any]) -> list[str]:
        """
        Extracts semantic identities from a collection of UI elements.
        """

        stable_identities = []
        interactive_elements = elements.values() if isinstance(elements, dict) else elements

        for element_info in interactive_elements:
            attributes = (
                element_info.get("attributes", {})
                if isinstance(element_info, dict)
                else getattr(element_info, "attributes", {})
            )

            if not attributes:
                continue

            class_name = str(attributes.get("class", "Unknown")).split(".")[-1]
            resource_id = str(attributes.get("resource-id", "")).split("/")[-1]

            element_text = str(attributes.get("text", "")).strip()[:30]
            element_description = str(attributes.get("content-desc", "")).strip()[:30]

            semantic_identity = f"{class_name}|{resource_id}|{element_text}|{element_description}"
            stable_identities.append(semantic_identity)

        return stable_identities
