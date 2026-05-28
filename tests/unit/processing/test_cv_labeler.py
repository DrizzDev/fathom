from __future__ import annotations

import io

from PIL import Image, ImageDraw

from fathom.processing.cv_labeler import VisualControlLabeler
from fathom.schemas.ui import LabeledElement, UIBounds


def _render_button_screenshot() -> bytes:
    """
    Render a black canvas with one saturated CTA rectangle to PNG bytes.
    """

    canvas = Image.new("RGB", (600, 900), "black")
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle((180, 420, 420, 500), radius=18, fill=(250, 95, 70))

    buffer = io.BytesIO()
    canvas.save(buffer, format="PNG")
    return buffer.getvalue()


def test_detects_uncovered_visual_button() -> None:
    """
    A rendered CTA missing from the hierarchy should be emitted as a
    provenance-marked visual control when consumed via in-memory bytes.
    """

    detected = VisualControlLabeler.detect(
        image=_render_button_screenshot(),
        existing_elements=[],
        scale_factor=1.0,
    )

    assert len(detected) == 1
    assert detected[0].attributes["source"] == "cv"
    assert detected[0].attributes["class"] == "VisualControl"
    assert detected[0].bounds.x1 <= 185
    assert detected[0].bounds.y1 <= 425
    assert detected[0].bounds.x2 >= 415
    assert detected[0].bounds.y2 >= 495


def test_skips_visual_control_already_covered_by_hierarchy() -> None:
    """
    Existing hierarchy coverage wins; CV labels should fill gaps, not
    duplicate already-labeled controls.
    """

    existing = [
        LabeledElement(
            label="1",
            color="red",
            bounds=UIBounds(x1=175, y1=415, x2=425, y2=505),
            attributes={"class": "Button", "label": "Already present"},
        )
    ]

    detected = VisualControlLabeler.detect(
        image=_render_button_screenshot(),
        existing_elements=existing,
        scale_factor=1.0,
    )

    assert detected == []


def test_returns_empty_when_image_bytes_are_empty() -> None:
    """
    Empty payload must short-circuit without attempting to decode.
    """

    detected = VisualControlLabeler.detect(
        image=b"",
        existing_elements=[],
        scale_factor=1.0,
    )

    assert detected == []


def test_returns_empty_when_image_bytes_are_undecodable() -> None:
    """
    Corrupt/undecodable payload must degrade gracefully.
    """

    detected = VisualControlLabeler.detect(
        image=b"not-a-real-png",
        existing_elements=[],
        scale_factor=1.0,
    )

    assert detected == []
