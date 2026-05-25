from __future__ import annotations

from enum import StrEnum


class ScrollEvidenceSource(StrEnum):
    """
    Source that contributed to a perceived scrollable region.
    """

    CORRELATION = "correlation"
    VERIFIER = "verifier"
    SURFACE = "surface"
