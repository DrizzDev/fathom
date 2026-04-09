from __future__ import annotations

import io
from logging import getLogger
from typing import Tuple

from PIL import Image

logger = getLogger(__name__)


def parse_png_dimensions(image: bytes) -> Tuple[int, int]:
    """Extract width and height from a PNG image's IHDR chunk.

    Lightweight alternative to decoding the full image with PIL.
    Raises ``ValueError`` when *image* is not a valid PNG or is
    truncated before the IHDR data.
    """

    if len(image) < 24:
        raise ValueError("Invalid PNG: payload too small to contain IHDR")

    if image[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("Invalid PNG: missing PNG signature")

    if image[12:16] != b"IHDR":
        raise ValueError("Invalid PNG: IHDR chunk not found at expected offset")

    width = int.from_bytes(image[16:20], byteorder="big")
    height = int.from_bytes(image[20:24], byteorder="big")
    return width, height


class ImageProcessor:
    """
    Utility for optimizing images for LLM processing.
    """

    @staticmethod
    def optimize_for_vision(
        image_data: bytes, max_dimension: int = 1024, quality: int = 80
    ) -> bytes:
        """
        Resizes and compresses an image to improve LLM latency.
        """
        try:
            with Image.open(io.BytesIO(image_data)) as image:
                # Calculate new dimensions maintaining aspect ratio
                width, height = image.size
                if width > max_dimension or height > max_dimension:
                    if width > height:
                        new_width = max_dimension
                        new_height = int(height * (max_dimension / width))
                    else:
                        new_height = max_dimension
                        new_width = int(width * (max_dimension / height))

                    image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

                # Compress
                buffer = io.BytesIO()
                image.convert("RGB").save(buffer, format="JPEG", quality=quality, optimize=True)
                return buffer.getvalue()
        except Exception as exception:
            logger.warning(f"Image optimization failed, using original: {exception}")
            return image_data
