from __future__ import annotations

import asyncio
from logging import getLogger
from typing import Any, Dict, List, Optional, Tuple

from fathom.constants.perception import VISUAL_CONTROL_MINIMUM_IOU
from fathom.core.exceptions import OcrError
from fathom.interfaces.ocr import OcrPort
from fathom.schemas.actions import Bounds
from fathom.schemas.budgets import PerceptionBudget
from fathom.schemas.ocr import OcrResult, OcrToken
from fathom.schemas.screens import ScreenCapture

logger = getLogger(__name__)


class OcrEnsembleService(OcrPort):
    """
    Multi-provider OCR aggregator that fans out to many members and merges tokens.
    """

    def __init__(
        self,
        *,
        members: Tuple[OcrPort, ...] = (),
        workflow_id: Optional[str] = None,
        deduplication_iou: float = VISUAL_CONTROL_MINIMUM_IOU,
    ) -> None:
        """
        Initialize the ensemble with members, dedupe threshold, and run context.
        """

        self.__members = members
        self.__workflow_id = workflow_id
        self.__deduplication_iou = deduplication_iou

    @property
    def members(self) -> Tuple[OcrPort, ...]:
        """
        Return the configured OCR members.
        """

        return self.__members

    async def extract(self, *, capture: ScreenCapture, budget: PerceptionBudget) -> OcrResult:
        """
        Fan out to every member and return a deduplicated, confidence-ranked result.
        """

        if not self.__members:
            return OcrResult(tokens=(), duration=0)

        context = self.__log_context(activity=capture.activity)
        logger.info(
            "OCR ensemble started",
            extra={
                **context,
                "event": "ocr.ensemble.started",
                "member.count": len(self.__members),
            },
        )

        outcomes = await asyncio.gather(
            *(
                self.__invoke_member(member=member, capture=capture, budget=budget)
                for member in self.__members
            ),
            return_exceptions=False,
        )

        merged = self.__merge(outcomes=[outcome for outcome in outcomes if outcome is not None])
        duration = max((outcome.duration for outcome in outcomes if outcome is not None), default=0)

        logger.info(
            "OCR ensemble completed",
            extra={
                **context,
                "duration.ms": duration,
                "token.count": len(merged),
                "event": "ocr.ensemble.completed",
            },
        )
        return OcrResult(tokens=merged, duration=duration)

    async def __invoke_member(
        self,
        *,
        member: OcrPort,
        capture: ScreenCapture,
        budget: PerceptionBudget,
    ) -> Optional[OcrResult]:
        """
        Call one member and isolate its failures from the rest of the ensemble.
        """

        context = self.__log_context(activity=capture.activity)

        try:
            return await member.extract(capture=capture, budget=budget)
        except OcrError as exception:
            logger.warning(
                "OCR ensemble member failed",
                extra={
                    **context,
                    "retryable": exception.retryable,
                    "error.message": exception.message,
                    "member.type": type(member).__name__,
                    "event": "ocr.ensemble.member.failed",
                },
            )
            return None
        except Exception as exception:
            logger.warning(
                "OCR ensemble member raised",
                extra={
                    **context,
                    "error.message": str(exception),
                    "event": "ocr.ensemble.member.error",
                    "member.type": type(member).__name__,
                },
            )
            return None

    def __merge(self, *, outcomes: List[OcrResult]) -> Tuple[OcrToken, ...]:
        """
        Combine multi-member token lists, keeping the highest-confidence per location.
        """

        if not outcomes:
            return ()

        ordered: List[OcrToken] = []

        for outcome in outcomes:
            ordered.extend(outcome.tokens)

        ordered.sort(key=lambda token: token.raw_score, reverse=True)

        survivors: List[OcrToken] = []

        for candidate in ordered:
            if any(
                survivor.text.casefold() == candidate.text.casefold()
                and self.__iou(first=survivor.bounds, second=candidate.bounds)
                >= self.__deduplication_iou
                for survivor in survivors
            ):
                continue

            survivors.append(candidate)

        return tuple(survivors)

    @staticmethod
    def __iou(*, first: Bounds, second: Bounds) -> float:
        """
        Return the intersection-over-union for two pixel bounds.
        """

        left = max(first.x, second.x)
        top = max(first.y, second.y)

        right = min(first.x + first.width, second.x + second.width)
        bottom = min(first.y + first.height, second.y + second.height)

        if right <= left or bottom <= top:
            return 0.0

        intersection = (right - left) * (bottom - top)

        first_area = first.width * first.height
        second_area = second.width * second.height

        union = first_area + second_area - intersection
        if union <= 0:
            return 0.0

        return float(intersection / union)

    def __log_context(self, *, activity: str) -> Dict[str, Any]:
        """
        Return shared structured-logging context for OCR ensemble entries.
        """

        return {
            "activity": activity,
            "workflow.id": self.__workflow_id,
            "component": "core.perception.ocr.ensemble",
        }
