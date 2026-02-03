from __future__ import annotations

import io
from logging import getLogger

from PIL import Image

logger = getLogger(__name__)


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
