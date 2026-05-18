from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from fathom.schemas.actions import Bounds
from fathom.schemas.budgets import PerceptionBudget
from fathom.schemas.screens import ScreenCapture


class OverlayDetectorPort(ABC):
    """
    Provider-neutral pixel-overlay detection contract consumed by perception.
    """

    @abstractmethod
    async def detect(self, *, capture: ScreenCapture, budget: PerceptionBudget) -> Optional[Bounds]:
        """
        Return the bounds of a detected dim/scrim overlay, or None when none is present.
        """

        raise NotImplementedError
