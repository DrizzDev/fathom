from __future__ import annotations

import io
import unittest
import xml.etree.ElementTree as ET  # nosec

from PIL import Image, ImageDraw

from fathom.constants import ActionType
from fathom.processing.drawer import BoundsGenerator


def _png_bytes(width: int = 1080, height: int = 2400) -> bytes:
    """
    Render a solid background of the requested size as PNG bytes.
    """

    canvas = Image.new("RGB", (width, height), "white")
    buffer = io.BytesIO()
    canvas.save(buffer, format="PNG")
    return buffer.getvalue()


def _png_with_saturated_button() -> bytes:
    """
    Render a black canvas with a saturated CTA the CV labeler will detect.
    """

    canvas = Image.new("RGB", (1080, 2400), "black")
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle((300, 1800, 780, 1900), radius=18, fill=(250, 95, 70))
    buffer = io.BytesIO()
    canvas.save(buffer, format="PNG")
    return buffer.getvalue()


_ANDROID_XML = """
<hierarchy package="com.test.app">
  <node class="android.widget.Button" bounds="[100,200][500,400]" clickable="true" text="Submit" />
  <node class="android.widget.TextView" bounds="[100,500][800,600]" clickable="false" text="Title" />
</hierarchy>
"""


class TestBoundsGeneratorBytesContract(unittest.TestCase):
    """
    Behavioural pins for the bytes-driven :meth:`BoundsGenerator.create_element` contract.
    """

    def test_create_element_returns_manifest_from_in_memory_bytes(self) -> None:
        """
        Bytes-only input must produce the labeled-element manifest with no
        filesystem dependency whatsoever.
        """

        root = ET.fromstring(_ANDROID_XML)

        elements, label_map = BoundsGenerator.create_element(
            root=root,
            image=_png_bytes(),
            action=ActionType.TAP,
            cv_enabled=False,
        )

        self.assertGreaterEqual(len(elements), 1)
        self.assertEqual(elements[0].label, "1")
        self.assertIn("1", label_map)
        self.assertIn("__scale_factor__", label_map)

    def test_create_element_raises_on_empty_bytes(self) -> None:
        """
        Empty payload is a programmer error and must fail fast at the boundary.
        """

        root = ET.fromstring(_ANDROID_XML)

        with self.assertRaises(ValueError):
            BoundsGenerator.create_element(
                root=root,
                image=b"",
                action=ActionType.TAP,
            )

    def test_create_element_returns_empty_when_payload_is_undecodable(self) -> None:
        """
        Undecodable bytes must degrade gracefully without raising.
        """

        root = ET.fromstring(_ANDROID_XML)

        elements, label_map = BoundsGenerator.create_element(
            root=root,
            image=b"not-a-real-png",
            action=ActionType.TAP,
        )

        self.assertEqual(elements, [])
        self.assertEqual(label_map, {})

    def test_create_element_invokes_cv_labeler_when_cv_enabled(self) -> None:
        """
        ``cv_enabled=True`` must route the same bytes payload into the
        visual-control labeler without resurrecting any filesystem dependency.
        """

        root = ET.fromstring(_ANDROID_XML)

        elements, label_map = BoundsGenerator.create_element(
            root=root,
            image=_png_bytes(),
            action=ActionType.TAP,
            cv_enabled=True,
        )

        self.assertGreaterEqual(len(elements), 1)
        self.assertIn("__scale_factor__", label_map)

    def test_create_element_appends_cv_visual_controls_when_detected(self) -> None:
        """
        When the CV labeler detects an uncovered saturated control, the
        bytes-driven branch must append it to the manifest alongside the
        XML-derived elements.
        """

        root = ET.fromstring(_ANDROID_XML)

        elements, label_map = BoundsGenerator.create_element(
            root=root,
            image=_png_with_saturated_button(),
            action=ActionType.TAP,
            cv_enabled=True,
        )

        sources = {entry.attributes.get("source") for entry in elements}
        self.assertIn("cv", sources)
        self.assertIn("__scale_factor__", label_map)
