from __future__ import annotations

import io
import unittest

from PIL import Image

from fathom.adapters.vision import PhashVisualHasher


def _solid_png(*, color: int, size: int = 64) -> bytes:
    """
    Build a one-color PNG of the given size for deterministic hashing.
    """

    image = Image.new("RGB", (size, size), color=(color, color, color))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class PhashVisualHasherTest(unittest.TestCase):
    """
    Cover the perceptual-hash adapter: deterministic, distinct for different inputs, robust to fallbacks.
    """

    def test_hash_is_stable_for_identical_bytes(self) -> None:
        """
        The same image bytes hash to the same digest across repeated calls.
        """

        hasher = PhashVisualHasher()
        bytes_in = _solid_png(color=64)
        first = hasher.hash(image=bytes_in)
        second = hasher.hash(image=bytes_in)
        self.assertEqual(first, second)
        self.assertGreater(len(first), 0)

    def test_hash_differs_for_distinct_images(self) -> None:
        """
        Visually distinct images produce different hashes.
        """

        hasher = PhashVisualHasher()
        gray = hasher.hash(image=_solid_png(color=128))
        black = hasher.hash(image=_solid_png(color=0))
        self.assertNotEqual(gray, black)

    def test_empty_bytes_does_not_raise(self) -> None:
        """
        Empty input still returns a string digest via the fallback path.
        """

        result = PhashVisualHasher().hash(image=b"")
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)
