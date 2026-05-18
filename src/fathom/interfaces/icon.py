from __future__ import annotations

from abc import ABC, abstractmethod

from fathom.schemas.budgets import PerceptionBudget
from fathom.schemas.icon import IconDetectionResult
from fathom.schemas.screens import ScreenCapture


class IconDetectorPort(ABC):
    """
    Provider-neutral icon-detection contract consumed by perception.
    """

    @abstractmethod
    async def detect(
        self, *, capture: ScreenCapture, budget: PerceptionBudget
    ) -> IconDetectionResult:
        """
        Detect known icons in the supplied screen capture.
        """

        raise NotImplementedError
