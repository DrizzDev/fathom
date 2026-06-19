from __future__ import annotations

import asyncio
from logging import getLogger
from typing import Any, Dict, List, Optional, Tuple

from fathom.constants.perception import ICON_NON_MAX_SUPPRESSION_IOU
from fathom.interfaces.icon import IconDetectorPort
from fathom.schemas.actions import Bounds
from fathom.schemas.budgets import PerceptionBudget
from fathom.schemas.icon import IconDetectionResult, IconMatch
from fathom.schemas.screens import ScreenCapture

logger = getLogger(__name__)


class IconEnsembleService(IconDetectorPort):
    """
    Multi-provider icon-detector aggregator that fans out and dedupes by IoU.
    """

    def __init__(
        self,
        *,
        workflow_id: Optional[str] = None,
        members: Tuple[IconDetectorPort, ...] = (),
        deduplication_iou: float = ICON_NON_MAX_SUPPRESSION_IOU,
    ) -> None:
        """
        Initialize the ensemble with members, dedupe threshold, and run context.
        """

        self.__members = members
        self.__workflow_id = workflow_id
        self.__deduplication_iou = deduplication_iou

    @property
    def members(self) -> Tuple[IconDetectorPort, ...]:
        """
        Return the configured icon-detector members.
        """

        return self.__members

    async def detect(
        self,
        *,
        capture: ScreenCapture,
        budget: PerceptionBudget,
    ) -> IconDetectionResult:
        """
        Fan out to every member and return a deduplicated, confidence-ranked result.
        """

        if not self.__members:
            return IconDetectionResult(matches=(), duration=0)

        context = self.__log_context(activity=capture.activity)
        logger.info(
            "Icon ensemble started",
            extra={
                **context,
                "event": "icon.ensemble.started",
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
            "Icon ensemble completed",
            extra={
                **context,
                "duration.ms": duration,
                "match.count": len(merged),
                "event": "icon.ensemble.completed",
            },
        )
        return IconDetectionResult(matches=merged, duration=duration)

    async def __invoke_member(
        self,
        *,
        capture: ScreenCapture,
        budget: PerceptionBudget,
        member: IconDetectorPort,
    ) -> Optional[IconDetectionResult]:
        """
        Call one member and isolate its failures from the rest of the ensemble.
        """

        context = self.__log_context(activity=capture.activity)

        try:
            return await member.detect(capture=capture, budget=budget)
        except Exception as exception:
            logger.warning(
                "Icon ensemble member raised",
                extra={
                    **context,
                    "error.message": str(exception),
                    "member.type": type(member).__name__,
                    "event": "icon.ensemble.member.error",
                },
            )
            return None

    def __merge(self, *, outcomes: List[IconDetectionResult]) -> Tuple[IconMatch, ...]:
        """
        Combine multi-member match lists, keeping the strongest per (kind, region).
        """

        if not outcomes:
            return ()

        ordered: List[IconMatch] = []

        for outcome in outcomes:
            ordered.extend(outcome.matches)

        ordered.sort(key=lambda match: match.confidence, reverse=True)

        survivors: List[IconMatch] = []

        for candidate in ordered:
            if any(
                survivor.kind == candidate.kind
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

        top = max(first.y, second.y)
        left = max(first.x, second.x)

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
        Return shared structured-logging context for icon ensemble entries.
        """

        return {
            "activity": activity,
            "workflow.id": self.__workflow_id,
            "component": "core.perception.icon.ensemble",
        }
