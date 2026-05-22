from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Awaitable, Callable, Optional, Tuple

from fathom.constants.scroll import ScrollDirection, ScrollVerdictKind
from fathom.schemas.actions import Bounds, GesturePath
from fathom.schemas.command import CommandPolicy
from fathom.schemas.configuration import ScrollInteractionPolicy
from fathom.schemas.observation import ScreenObservation
from fathom.schemas.results import ActionTraceEvent
from fathom.schemas.screens import ScreenCapture
from fathom.schemas.scroll import (
    ScrollAttempt,
    ScrollContext,
    ScrollScope,
    ScrollSurface,
    ScrollVerdict,
)
from fathom.utils.coordinates import CoordinateConverter


class ScrollDetectPort(ABC):
    """
    Port that deterministically evaluates one attempted scroll.
    """

    @abstractmethod
    async def evaluate(
        self,
        *,
        before: ScreenCapture,
        after: ScreenCapture,
        region: Bounds,
        direction: ScrollDirection,
    ) -> ScrollVerdict:
        """
        Classify what happened inside the observed capture region.
        """

        raise NotImplementedError


class ScrollVerifyPort(ABC):
    """
    Port that resolves ambiguous scroll observations.
    """

    @abstractmethod
    async def verify(
        self,
        *,
        before: ScreenCapture,
        after: ScreenCapture,
        region: Bounds,
        direction: ScrollDirection,
    ) -> ScrollVerdict:
        """
        Resolve an ambiguous observation using a slower but richer signal.
        """

        raise NotImplementedError


class ScrollSurfacePort(ABC):
    """
    Port that extracts scroll-interfering surface hints.
    """

    @abstractmethod
    async def inspect(
        self,
        *,
        observation: ScreenObservation,
        path: GesturePath,
        capture_width: int,
        capture_height: int,
    ) -> Tuple[ScrollSurface, ...]:
        """
        Return surface hints that may interfere with the proposed gesture.
        """

        raise NotImplementedError


class ScrollRuntimePolicyPort(ABC):
    """
    Port that owns scroll runtime decisions shared by the supervisor.
    """

    @abstractmethod
    def requested_scope(
        self,
        *,
        context: ScrollContext,
        converter: CoordinateConverter,
    ) -> ScrollScope:
        """
        Build the scope implied by the original request when no explicit container resolves.
        """

        raise NotImplementedError

    @abstractmethod
    def initial_verdict(self) -> ScrollVerdict:
        """
        Build the initial unresolved verdict.
        """

        raise NotImplementedError

    @abstractmethod
    def budget_verdict(self) -> ScrollVerdict:
        """
        Build the budget-exhausted verdict.
        """

        raise NotImplementedError

    @abstractmethod
    def device_failure_verdict(self, *, error: str | None) -> ScrollVerdict:
        """
        Build the device-failure verdict.
        """

        raise NotImplementedError

    @abstractmethod
    def is_success(self, *, verdict: ScrollVerdict) -> bool:
        """
        Return whether one verdict confirms useful movement.
        """

        raise NotImplementedError

    @abstractmethod
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

        raise NotImplementedError

    @abstractmethod
    def maximum_internal_attempts(self, *, policy: CommandPolicy) -> int:
        """
        Return the bounded number of in-execute attempts allowed for one scroll action.
        """

        raise NotImplementedError


class ScrollPlanPort(ABC):
    """
    Port that builds bounded scroll attempts inside one resolved scope.
    """

    @abstractmethod
    def plan(
        self,
        *,
        context: ScrollContext,
        current: GesturePath,
        scope: Optional[ScrollScope],
        surfaces: Tuple[ScrollSurface, ...],
        converter: CoordinateConverter,
        policy: ScrollInteractionPolicy.AdaptivePolicy,
        capture_height: int,
    ) -> Tuple[ScrollAttempt, ...]:
        """
        Return ordered attempts for one scoped scroll execution.
        """

        raise NotImplementedError

    @abstractmethod
    def log_attempt(self, *, attempt: ScrollAttempt, attempt_index: int) -> None:
        """
        Record one dispatched scroll attempt for observability.
        """

        raise NotImplementedError

    @abstractmethod
    def ambiguous_verdict_kind(self) -> ScrollVerdictKind:
        """
        Return the verdict kind representing an ambiguous outcome.
        """

        raise NotImplementedError


TraceRecorder = Callable[[ActionTraceEvent], Awaitable[None]]
