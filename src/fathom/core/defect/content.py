"""
Content defect detection from a screen's text, without a model call.
"""

from __future__ import annotations

import re
from typing import List, Pattern, Set, Tuple

from fathom.constants.defect import PLACEHOLDER_SIGNALS, DefectSignal, DefectSource
from fathom.interfaces.defect import ScreenDefectDetectorPort
from fathom.schemas.defect import Defect, DefectEvidence, ScreenSnapshot

# Placeholder markers precompiled as word-boundary patterns so "todoist" does
# not trip "todo"; ordered longest-first so "lorem ipsum" wins over "lorem".
_PLACEHOLDER_PATTERNS: Tuple[Tuple[str, Pattern[str], DefectSignal], ...] = tuple(
    (marker, re.compile(rf"\b{re.escape(marker)}\b", re.IGNORECASE), signal)
    for marker, signal in PLACEHOLDER_SIGNALS.items()
)


class ContentDefectDetector(ScreenDefectDetectorPort):
    """
    Flags unfinished or placeholder copy in a screen's text, without a model.

    Scans the model's own description of the screen, so it catches blatant
    placeholder copy (lorem ipsum, TODO) but cannot see text the description
    paraphrases away. A vision pass over the screenshot is the reliable
    complement.
    """

    async def inspect_screen(self, *, snapshot: ScreenSnapshot) -> List[Defect]:
        """
        Returns one content defect per distinct placeholder marker on the screen.
        """

        corpus = "\n".join(text for text in snapshot.texts if text)
        if not corpus.strip():
            return []

        defects: List[Defect] = []
        seen: Set[DefectSignal] = set()
        for marker, pattern, signal in _PLACEHOLDER_PATTERNS:
            if signal in seen or not pattern.search(corpus):
                continue
            seen.add(signal)
            defects.append(
                Defect.from_signal(
                    signal=signal,
                    source=DefectSource.POST_RUN,
                    summary=f"Screen text contains placeholder copy ('{marker}')",
                    evidence=DefectEvidence(
                        screen=snapshot.screen,
                        activity=snapshot.activity,
                        excerpt=marker,
                    ),
                )
            )
        return defects
