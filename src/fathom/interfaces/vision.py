from __future__ import annotations

from abc import ABC, abstractmethod


class VisualHasher(ABC):
    """
    Computes a perceptual hash from raw screen image bytes.
    """

    @abstractmethod
    def hash(self, *, image: bytes) -> str:
        """
        Return a stable hex-encoded perceptual hash for the given image bytes.
        """

        raise NotImplementedError
