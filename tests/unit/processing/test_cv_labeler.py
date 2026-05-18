from __future__ import annotations

from typing import TYPE_CHECKING

from PIL import Image, ImageDraw

from fathom.processing.cv_labeler import VisualControlLabeler
from fathom.schemas.ui import LabeledElement, UIBounds

if TYPE_CHECKING:
    from pathlib import Path


def test_detects_uncovered_visual_button(tmp_path: Path) -> None:
    """
    A rendered CTA missing from the hierarchy should be emitted as a
    provenance-marked visual control.
    """

    image_path = tmp_path / "screen.png"
    image = Image.new("RGB", (600, 900), "black")
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((180, 420, 420, 500), radius=18, fill=(250, 95, 70))
    image.save(image_path)

    detected = VisualControlLabeler.detect(
        image_path=image_path,
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


def test_skips_visual_control_already_covered_by_hierarchy(tmp_path: Path) -> None:
    """
    Existing hierarchy coverage wins; CV labels should fill gaps, not
    duplicate already-labeled controls.
    """

    image_path = tmp_path / "screen.png"
    image = Image.new("RGB", (600, 900), "black")
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((180, 420, 420, 500), radius=18, fill=(250, 95, 70))
    image.save(image_path)

    existing = [
        LabeledElement(
            label="1",
            color="red",
            bounds=UIBounds(x1=175, y1=415, x2=425, y2=505),
            attributes={"class": "Button", "label": "Already present"},
        )
    ]

    detected = VisualControlLabeler.detect(
        image_path=image_path,
        existing_elements=existing,
        scale_factor=1.0,
    )

    assert detected == []
