from __future__ import annotations

from typing import Optional

from fathom.interfaces.overlay import OverlayDetectorPort
from fathom.schemas.actions import Bounds
from fathom.schemas.budgets import PerceptionBudget
from fathom.schemas.screens import ScreenCapture


class NoopOverlayDetector(OverlayDetectorPort):
    """
    Overlay detector that intentionally returns no overlay.
    """

    async def detect(
        self,
        *,
        capture: ScreenCapture,
        budget: PerceptionBudget,
    ) -> Optional[Bounds]:
        """
        Return None without consulting any pixel-level evidence.
        """

        _ = capture
        _ = budget
        return None
