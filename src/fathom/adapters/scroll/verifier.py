from __future__ import annotations

from io import BytesIO
from typing import Optional

from PIL import Image

from fathom.constants.scroll import (
    ScrollDirection,
    ScrollEvidenceSource,
    ScrollVerdictKind,
)
from fathom.interfaces.llm import LLMPort
from fathom.interfaces.scroll import ScrollVerifyPort
from fathom.schemas.actions import Bounds
from fathom.schemas.screens import ScreenCapture
from fathom.schemas.scroll import ScrollVerdict


class LlmScrollVerifier(ScrollVerifyPort):
    """
    Narrow verifier for ambiguous scroll observations.
    """

    __SYSTEM = (
        "You compare two crops of the same mobile screen region before and after one swipe. "
        "Answer with exactly one token: PROGRESSED, NO_PROGRESS, WRONG_AXIS, or AMBIGUOUS. "
        "PROGRESSED means the same content moved in the requested direction. "
        "NO_PROGRESS means the region did not move meaningfully. "
        "WRONG_AXIS means movement happened on another axis. "
        "AMBIGUOUS means the crops are not decisive."
    )

    def __init__(self, *, llm: LLMPort) -> None:
        """
        Bind verifier to an injected LLM port.
        """

        self.__llm = llm

    async def verify(
        self,
        *,
        before: ScreenCapture,
        after: ScreenCapture,
        region: Bounds,
        direction: ScrollDirection,
    ) -> ScrollVerdict:
        """
        Resolve an ambiguous deterministic observation.
        """

        before_crop = self.__crop(capture=before, region=region)
        after_crop = self.__crop(capture=after, region=region)
        if before_crop is None or after_crop is None:
            return self.__verdict(
                kind=ScrollVerdictKind.AMBIGUOUS,
                confidence=0.0,
                detail="crop_failed",
            )

        result = await self.__llm.generate(
            use_cache=False,
            system_instruction=self.__SYSTEM,
            prompt=[
                f"Expected direction: {direction.value}.",
                "Before:",
                before_crop,
                "After:",
                after_crop,
                "Answer with exactly one token: PROGRESSED, NO_PROGRESS, WRONG_AXIS, or AMBIGUOUS.",
            ],
        )
        return self.__parse(content=result.content)

    @staticmethod
    def __parse(*, content: str) -> ScrollVerdict:
        """
        Parse the verifier token into a verdict.
        """

        token = content.strip().upper().split()[0] if content else ""
        try:
            kind = ScrollVerdictKind(token.lower())
        except ValueError:
            return LlmScrollVerifier.__verdict(
                kind=ScrollVerdictKind.AMBIGUOUS,
                confidence=0.0,
                detail="unparseable_response",
            )

        return LlmScrollVerifier.__verdict(
            kind=kind,
            confidence=0.95,
            detail="llm_verified",
        )

    @staticmethod
    def __crop(*, capture: ScreenCapture, region: Bounds) -> Optional[bytes]:
        """
        Crop one capture to PNG bytes.
        """

        try:
            image = Image.open(BytesIO(capture.image))
        except Exception:
            return None

        left = max(0, region.x)
        top = max(0, region.y)
        right = min(image.width, region.x + region.width)
        bottom = min(image.height, region.y + region.height)
        if right <= left or bottom <= top:
            return None

        buffer = BytesIO()
        image.crop((left, top, right, bottom)).save(buffer, format="PNG")
        return buffer.getvalue()

    @staticmethod
    def __verdict(
        *,
        kind: ScrollVerdictKind,
        confidence: float,
        detail: Optional[str],
    ) -> ScrollVerdict:
        """
        Build a verifier verdict.
        """

        return ScrollVerdict(
            kind=kind,
            source=ScrollEvidenceSource.VERIFIER,
            confidence=max(0.0, min(1.0, confidence)),
            distance=0,
            detail=detail,
        )
