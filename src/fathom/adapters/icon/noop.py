from __future__ import annotations

from fathom.interfaces.icon import IconDetectorPort
from fathom.schemas.budgets import PerceptionBudget
from fathom.schemas.icon import IconDetectionResult
from fathom.schemas.screens import ScreenCapture


class NoopIconDetector(IconDetectorPort):
    """
    Icon detector that intentionally returns no matches.
    """

    async def detect(
        self, *, capture: ScreenCapture, budget: PerceptionBudget
    ) -> IconDetectionResult:
        """
        Return an empty icon-detection result without consulting any provider.
        """

        _ = capture
        _ = budget

        return IconDetectionResult(matches=(), duration=0)
