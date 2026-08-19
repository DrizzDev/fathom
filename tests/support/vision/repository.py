from __future__ import annotations

import io
from pathlib import Path
from typing import Tuple

from PIL import Image

from tests.fixtures.vision.corpus import VisionCase
from tests.support.vision.models import EvaluationReport


class VisionCaseRepository:
    """
    Loads real-pixel case bytes and dimensions; the case entities stay data-only.
    """

    def __init__(self, *, root: Path | None = None) -> None:
        """
        Bind the fixtures root; defaults to the committed tests/fixtures/vision directory.
        """

        self.__root = (
            root
            if root is not None
            else Path(__file__).resolve().parents[2] / "fixtures" / "vision"
        )

    @property
    def screens(self) -> Path:
        """
        Directory holding the committed screenshots.
        """

        return self.__root / "screens"

    def image_bytes(self, *, case: VisionCase) -> bytes:
        """
        Return the committed pixels for a case.
        """

        return (self.screens / case.screenshot).read_bytes()

    def dimensions(self, *, case: VisionCase) -> Tuple[int, int]:
        """
        Return the pixel width and height of a case screenshot.
        """

        with Image.open(io.BytesIO(self.image_bytes(case=case))) as image:
            return image.size


class ReportWriter:
    """
    Persists an evaluation report as canonical JSON.
    """

    def write(self, *, report: EvaluationReport, path: Path) -> None:
        """
        Serialize the typed report to the given path, creating parent directories.
        """

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(report.model_dump_json(indent=2))
