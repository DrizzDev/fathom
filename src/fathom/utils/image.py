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
            with Image.open(io.BytesIO(image_data)) as opened:
                width, height = opened.size
                needs_resize = width > max_dimension or height > max_dimension

                if needs_resize:
                    if width > height:
                        new_width = max_dimension
                        new_height = int(height * (max_dimension / width))
                    else:
                        new_height = max_dimension
                        new_width = int(width * (max_dimension / height))
                    resized = opened.resize((new_width, new_height), Image.Resampling.LANCZOS)
                else:
                    resized = None

                source = resized if resized is not None else opened
                try:
                    converted = source.convert("RGB")
                    try:
                        buffer = io.BytesIO()
                        converted.save(buffer, format="JPEG", quality=quality, optimize=True)
                        return buffer.getvalue()
                    finally:
                        converted.close()
                finally:
                    if resized is not None:
                        resized.close()
        except Exception as exception:
            logger.warning(f"Image optimization failed, using original: {exception}")
            return image_data
