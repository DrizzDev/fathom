from __future__ import annotations

from typing import List, Optional, Protocol

from fathom.schemas.screens import ScreenCapture, ScreenDiff, ScreenHashBundle, ScreenState
from fathom.schemas.ui import LabeledElement


class ScreenStatePort(Protocol):
    """
    Builds comparable screen state from captured screens.
    """

    def resolve_capture_hashes(
        self, *, capture: ScreenCapture, elements: List[LabeledElement]
    ) -> ScreenHashBundle:
        """
        Resolve visual, XML, and interaction hashes for a capture.
        """
        ...

    def build_screen_state(
        self,
        *,
        xml_hash: str,
        visual_hash: str,
        interaction_hash: str,
        capture: ScreenCapture,
    ) -> ScreenState:
        """
        Build comparable screen state for a capture.
        """
        ...


class ScreenComparisonPort(Protocol):
    """
    Compares before and after screen captures.
    """

    def compare(
        self,
        *,
        after: ScreenCapture,
        before: ScreenCapture,
        after_state: ScreenState,
        before_state: Optional[ScreenState],
    ) -> ScreenDiff:
        """
        Compare before and after captures.
        """
        ...
