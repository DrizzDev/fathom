from __future__ import annotations

from fathom.interfaces.ocr import OcrPort
from fathom.schemas.budgets import PerceptionBudget
from fathom.schemas.ocr import OcrResult
from fathom.schemas.screens import ScreenCapture


class NoopOcr(OcrPort):
    """
    OCR adapter that intentionally returns no tokens.
    """

    async def extract(self, *, capture: ScreenCapture, budget: PerceptionBudget) -> OcrResult:
        """
        Return an empty OCR result without consulting any external provider.
        """

        _ = capture
        _ = budget

        return OcrResult(tokens=(), duration=0)
