from __future__ import annotations

import time

from fathom.constants.command import CommandScopeKind
from fathom.constants.scroll import ScrollDirection, ScrollEvidenceSource, ScrollVerdictKind
from fathom.interfaces.scroll import ScrollRuntimePolicyPort
from fathom.schemas.actions import Bounds, CoordinateSource, CoordinateSystem
from fathom.schemas.command import CommandPolicy
from fathom.schemas.scroll import ScrollContext, ScrollScope, ScrollVerdict
from fathom.utils.coordinates import CoordinateConverter


class ScrollRuntimePolicy(ScrollRuntimePolicyPort):
    """
    Owns bounded runtime decisions for supervised scroll execution.
    """

    def requested_scope(
        self,
        *,
        context: ScrollContext,
        converter: CoordinateConverter,
    ) -> ScrollScope:
        """
        Build the scope implied by the original request.
        """

        bounds = converter.capture_bounds(region=context.region)
        return ScrollScope(
            identifier="requested_scope",
            kind=CommandScopeKind.VIEWPORT,
            bounds=Bounds(
                x=bounds.x,
                y=bounds.y,
                width=bounds.width,
                height=bounds.height,
                coordinate_system=CoordinateSystem.DEVICE_PIXEL,
                source=CoordinateSource.VIEWPORT,
            ),
            region=context.region,
            axis=(
                "vertical"
                if context.direction in {ScrollDirection.UP, ScrollDirection.DOWN}
                else "horizontal"
            ),
            confidence=0.30,
            source=ScrollEvidenceSource.SURFACE,
            manifest_label_id=context.anchor.manifest_label_id,
            observation_region_id=context.anchor.observation_region_id,
        )

    def initial_verdict(self) -> ScrollVerdict:
        """
        Build the initial unresolved verdict.
        """

        return self.__ambiguous_verdict(detail="no_attempt_executed")

    def budget_verdict(self) -> ScrollVerdict:
        """
        Build the budget-exhausted verdict.
        """

        return self.__ambiguous_verdict(detail="scroll_budget_exhausted")

    def device_failure_verdict(self, *, error: str | None) -> ScrollVerdict:
        """
        Build the device-failure verdict.
        """

        return self.__ambiguous_verdict(detail=error or "device_swipe_failed")

    def is_success(self, *, verdict: ScrollVerdict) -> bool:
        """
        Return whether one verdict confirms useful movement.
        """

        return verdict.kind is ScrollVerdictKind.PROGRESSED

    def should_continue(
        self,
        *,
        verdict: ScrollVerdict,
        completed_count: int,
        policy: CommandPolicy,
        deadline: float,
    ) -> bool:
        """
        Return whether another scoped retry is justified.
        """

        if completed_count >= policy.attempts:
            return False
        if time.time() >= deadline:
            return False
        if verdict.kind is ScrollVerdictKind.AMBIGUOUS:
            return True
        if verdict.kind in {ScrollVerdictKind.NO_PROGRESS, ScrollVerdictKind.WRONG_AXIS}:
            return completed_count < min(2, policy.attempts)
        return False

    def maximum_internal_attempts(self, *, policy: CommandPolicy) -> int:
        """
        Return the maximum number of in-execute attempts for one scroll action.
        """

        return 1

    @staticmethod
    def __ambiguous_verdict(*, detail: str) -> ScrollVerdict:
        """
        Build one ambiguous verdict with a stable default payload.
        """

        return ScrollVerdict(
            kind=ScrollVerdictKind.AMBIGUOUS,
            source=ScrollEvidenceSource.SURFACE,
            confidence=0.0,
            distance=0,
            detail=detail,
        )
