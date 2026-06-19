from __future__ import annotations

import io
import unittest

from PIL import Image

from fathom.processing.annotator import ImageAnnotator
from fathom.schemas.ui import LabeledElement, UIBounds


def _solid_png(width: int = 600, height: int = 800) -> bytes:
    """
    Produce solid-coloured PNG bytes for annotator round-trips.
    """

    canvas = Image.new("RGB", (width, height), "black")
    buffer = io.BytesIO()
    canvas.save(buffer, format="PNG")
    return buffer.getvalue()


class TestImageAnnotatorBytesContract(unittest.TestCase):
    """
    Behavioural pins for the bytes-in / bytes-out :meth:`ImageAnnotator.annotate` contract.
    """

    def test_annotate_returns_png_bytes_for_valid_input(self) -> None:
        """
        A successful annotation must yield decodable PNG bytes whose
        dimensions match the source canvas.
        """

        source = _solid_png(width=600, height=800)
        elements = [
            LabeledElement(
                label="1",
                color="#FF3B30",
                bounds=UIBounds(x1=100, y1=120, x2=300, y2=200),
                attributes={"class": "Button"},
            )
        ]

        annotated = ImageAnnotator.annotate(image=source, elements=elements)

        self.assertIsNotNone(annotated)
        with Image.open(io.BytesIO(annotated)) as decoded:
            self.assertEqual(decoded.width, 600)
            self.assertEqual(decoded.height, 800)
            self.assertEqual(decoded.format, "PNG")

    def test_annotate_handles_empty_element_list(self) -> None:
        """
        An empty element list must still return the encoded canvas, not None.
        """

        source = _solid_png()
        annotated = ImageAnnotator.annotate(image=source, elements=[])

        self.assertIsNotNone(annotated)
        with Image.open(io.BytesIO(annotated)) as decoded:
            self.assertEqual(decoded.format, "PNG")

    def test_annotate_raises_on_empty_bytes(self) -> None:
        """
        Empty payload at the boundary is a programmer error and must fail fast.
        """

        with self.assertRaises(ValueError):
            ImageAnnotator.annotate(image=b"", elements=[])

    def test_annotate_returns_none_on_undecodable_bytes(self) -> None:
        """
        Undecodable payload must degrade by returning None so the caller
        can fall back without crashing the hierarchy stage.
        """

        annotated = ImageAnnotator.annotate(image=b"not-a-real-png", elements=[])

        self.assertIsNone(annotated)
