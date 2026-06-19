from __future__ import annotations

import io
from logging import getLogger
from typing import Tuple

from PIL import Image, UnidentifiedImageError

logger = getLogger(__name__)


class CaptureOrientationResolver:
    """
    Aligns the logical capture dimensions with the screenshot's actual aspect.
    """

    @classmethod
    def resolve(
        cls,
        *,
        image: bytes,
        reported_width: int,
        reported_height: int,
    ) -> Tuple[int, int]:
        """
        Return logical dimensions whose orientation matches the supplied image.
        """

        if reported_width <= 0 or reported_height <= 0:
            return reported_width, reported_height

        decoded = cls.__decode(image=image)
        if decoded is None:
            return reported_width, reported_height

        image_width, image_height = decoded
        if image_width == image_height:
            return reported_width, reported_height

        image_is_landscape = image_width > image_height
        reported_is_landscape = reported_width > reported_height

        if image_is_landscape == reported_is_landscape:
            return reported_width, reported_height

        logger.info(
            "Capture orientation corrected against image aspect",
            extra={
                "component": "core.perception.orientation",
                "event": "capture.orientation.corrected",
                "image.width": image_width,
                "image.height": image_height,
                "reported.width": reported_width,
                "reported.height": reported_height,
                "corrected.width": reported_height,
                "corrected.height": reported_width,
            },
        )
        return reported_height, reported_width

    @staticmethod
    def __decode(*, image: bytes) -> Tuple[int, int] | None:
        """
        Read the screenshot's pixel dimensions from the PNG header.
        """

        if not image:
            return None
        try:
            with Image.open(io.BytesIO(image)) as decoded:
                return decoded.width, decoded.height
        except (OSError, ValueError, UnidentifiedImageError):
            return None
