from __future__ import annotations

from abc import ABC, abstractmethod

from fathom.schemas.budgets import PerceptionBudget
from fathom.schemas.ocr import OcrResult
from fathom.schemas.screens import ScreenCapture


class OcrPort(ABC):
    """
    Provider-neutral OCR contract consumed by perception and supervision.
    """

    @abstractmethod
    async def extract(self, *, capture: ScreenCapture, budget: PerceptionBudget) -> OcrResult:
        """
        Extract OCR tokens from the supplied screen capture.
        """

        raise NotImplementedError
