from __future__ import annotations

import hashlib
import io
from logging import getLogger
from typing import List

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
from fathom.core.exceptions import MissingDependencyError, VisionError

logger = getLogger(__name__)


class VisualHashEngine:
    """
    Stateless engine that computes a perceptual hash from raw image bytes.
    """

    def hash(self, *, image: bytes) -> str:
        """
        Return a 64-bit hex perceptual hash for the given image bytes; falls back to SHA256 on failure.
        """

        try:
            if OPENCV_AVAILABLE:
                return self.__compute_phash_opencv(image_data=image)
            return self.__compute_phash_pillow(image_data=image)
        except Exception as exception:
            logger.warning(f"Could not compute pHash, falling back to SHA256: {exception}")
            return hashlib.sha256(image).hexdigest()[:VISUAL_HASH_LENGTH]

    def __compute_phash_opencv(self, *, image_data: bytes) -> str:
        """
        Compute pHash using OpenCV + NumPy (primary path).
        """

        if not OPENCV_AVAILABLE or cv2 is None or numpy is None:
            raise MissingDependencyError(dependency="opencv-python", feature="pHash computation")

        logger.debug("Computing pHash using OpenCV")

        image_array = numpy.frombuffer(image_data, numpy.uint8)
        decoded_image = cv2.imdecode(image_array, cv2.IMREAD_GRAYSCALE)

        if decoded_image is None:
            raise VisionError("Could not decode image with OpenCV")

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
            pixel_data: List[int] = []

            for pixel_value in grayscale_image.getdata():
                if isinstance(pixel_value, int):
                    pixel_data.append(pixel_value)
                elif isinstance(pixel_value, float):
                    pixel_data.append(int(pixel_value))
                elif isinstance(pixel_value, (tuple, list)) and len(pixel_value) > 0:
                    pixel_data.append(int(pixel_value[0]))
                else:
                    pixel_data.append(0)

            hash_integer = 0

            for row_index in range(8):
                for col_index in range(8):
                    pixel_index = row_index * 9 + col_index
                    if pixel_data[pixel_index] > pixel_data[pixel_index + 1]:
                        hash_integer |= 1 << (63 - (row_index * 8 + col_index))

            return f"{hash_integer:016x}"
