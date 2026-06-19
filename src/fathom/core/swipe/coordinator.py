from __future__ import annotations

from logging import getLogger
from typing import List, Optional, Tuple

from fathom.constants.swipe import ABORT_REASON_PRECEDENCE, AbortReason
from fathom.core.swipe.planner import SwipeRetryPlanner
from fathom.interfaces.swipe import SwipeAttemptDispatcher
from fathom.schemas.actions import Bounds, GesturePath
from fathom.schemas.observation import KeyboardObservation
from fathom.schemas.swipe import (
    SwipeAttempt,
    SwipeExecution,
    SwipeRejection,
    SwipeRetryPolicy,
)

logger = getLogger(__name__)


class SwipeRetryCoordinator:
    """
    Application service that orchestrates one logical swipe with bounded coordinate retries.
    """

    def __init__(
        self,
        *,
        planner: SwipeRetryPlanner,
        dispatcher: SwipeAttemptDispatcher,
    ) -> None:
        """
        Bind the coordinator to its pure planner and dispatch port.
        """

        self.__planner = planner
        self.__dispatcher = dispatcher

    async def execute(
        self,
        *,
        original: GesturePath,
        bounds: Bounds,
        policy: SwipeRetryPolicy,
        keyboard: KeyboardObservation,
        original_before: str,
    ) -> SwipeExecution:
        """
        Plan candidates, dispatch until visual change or candidates exhaust, return typed execution.
        """

        sequence = self.__planner.candidates(
            original=original,
            bounds=bounds,
            policy=policy,
            keyboard=keyboard,
        )

        attempts: List[SwipeAttempt] = []
        last_path: Optional[GesturePath] = None
        succeeded = False

        for index, path in enumerate(sequence.accepted):
            attempt = await self.__dispatcher.attempt(
                path=path,
                index=index,
                original_before=original_before,
            )
            attempts.append(attempt)
            last_path = path
            if attempt.device.succeeded and attempt.visual.changed:
                succeeded = True
                break

        aborted_for = self.__resolve_abort_reason(
            succeeded=succeeded,
            attempts=tuple(attempts),
            rejections=sequence.rejections,
        )

        return SwipeExecution(
            attempts=tuple(attempts),
            rejections=sequence.rejections,
            final=last_path,
            aborted_for=aborted_for,
        )

    @staticmethod
    def __resolve_abort_reason(
        *,
        succeeded: bool,
        attempts: Tuple[SwipeAttempt, ...],
        rejections: Tuple[SwipeRejection, ...],
    ) -> Optional[AbortReason]:
        """
        Resolve the abort reason using documented precedence when execution did not produce visual change.
        """

        if succeeded:
            return None

        observed_reasons: set[AbortReason] = set()
        for rejection in rejections:
            observed_reasons.add(rejection.reason)
        for attempt in attempts:
            if attempt.device.succeeded and attempt.visual.after is None:
                observed_reasons.add(AbortReason.CAPTURE_FAILED)
            elif not attempt.device.succeeded:
                observed_reasons.add(AbortReason.DEVICE_FAILED)
            elif not attempt.visual.changed:
                observed_reasons.add(AbortReason.NO_VISUAL_CHANGE)

        for candidate in ABORT_REASON_PRECEDENCE:
            if candidate in observed_reasons:
                return candidate
        return None
