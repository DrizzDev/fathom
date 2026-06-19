from __future__ import annotations

from fathom.core.perception.hashing import VisualHashEngine
from fathom.interfaces.vision import VisualHasher


class PhashVisualHasher(VisualHasher):
    """
    Adapter that exposes :class:`VisualHashEngine` through the :class:`VisualHasher` port.
    """

    def __init__(self, *, engine: VisualHashEngine | None = None) -> None:
        """
        Bind the adapter to an injected hashing engine, defaulting to the shared implementation.
        """

        self.__engine = engine or VisualHashEngine()

    def hash(self, *, image: bytes) -> str:
        """
        Delegate to the underlying engine to compute a perceptual hash for the image bytes.
        """

        return self.__engine.hash(image=image)
