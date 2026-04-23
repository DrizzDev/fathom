from typing import TYPE_CHECKING

from fathom.utils.coordinates import CoordinateConverter

if TYPE_CHECKING:
    from fathom.utils.image import ImageProcessor

__all__ = [
    "ImageProcessor",
    "CoordinateConverter",
]


def __getattr__(name: str) -> object:
    """
    Lazily expose optional utility exports without importing their dependencies.
    """

    if name == "ImageProcessor":
        from fathom.utils.image import ImageProcessor

        return ImageProcessor

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
