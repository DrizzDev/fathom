from __future__ import annotations

from threading import Lock
from typing import List, Optional

from fathom.constants.llm import (
    InferencePriorityMode,
    InferencePriorityTransitionReason,
    InferenceTier,
)
from fathom.schemas.configuration import PriorityInferenceConfiguration
from fathom.schemas.llm import (
    PriorityInferenceEvidence,
    PriorityInferenceSignal,
    PriorityInferenceTransition,
)


class PriorityInferencePolicy:
    """
    Selects provider-neutral inference tier from configuration and recent outcomes.
    """

    def __init__(self, *, configuration: PriorityInferenceConfiguration) -> None:
        """
        Initialize the selector with immutable configuration and empty state.
        """

        self.__configuration = configuration
        self.__signals: List[PriorityInferenceSignal] = []

        self.__healthy = 0
        self.__elevated = False

        self.__lock = Lock()

    def select(self) -> InferenceTier:
        """
        Return the tier to use for the next LLM attempt.
        """

        if not self.__configuration.enabled:
            return InferenceTier.STANDARD

        if self.__configuration.mode == InferencePriorityMode.ALWAYS:
            return InferenceTier.PRIORITY

        with self.__lock:
            if self.__elevated:
                return InferenceTier.PRIORITY

        return InferenceTier.STANDARD

    def record(self, *, signal: PriorityInferenceSignal) -> Optional[PriorityInferenceTransition]:
        """
        Update adaptive state and return a transition when the selected tier changes.
        """

        if (
            not self.__configuration.enabled
            or self.__configuration.mode != InferencePriorityMode.ADAPTIVE
        ):
            return None

        with self.__lock:
            self.__remember(signal=signal)

            if self.__elevated:
                return self.__recover(signal=signal)

            reason = self.__elevation_reason()

            if reason is not None:
                evidence = self.__evidence()

                self.__healthy = 0
                self.__elevated = True

                return PriorityInferenceTransition(
                    reason=reason,
                    evidence=evidence,
                    current=InferenceTier.PRIORITY,
                    previous=InferenceTier.STANDARD,
                )

            return None

    def __remember(self, *, signal: PriorityInferenceSignal) -> None:
        """
        Store the most recent bounded signal history.
        """

        self.__signals.append(signal)
        window = self.__configuration.adaptive.window

        if len(self.__signals) > window:
            self.__signals = self.__signals[-window:]

    def __recover(
        self, *, signal: PriorityInferenceSignal
    ) -> Optional[PriorityInferenceTransition]:
        """
        Scale down after enough consecutive healthy priority attempts.
        """

        if self.__is_healthy_priority(signal=signal):
            self.__healthy += 1
        else:
            self.__healthy = 0

        if self.__healthy >= self.__configuration.adaptive.threshold.recovery:
            evidence = self.__evidence()

            self.__healthy = 0
            self.__signals.clear()
            self.__elevated = False

            return PriorityInferenceTransition(
                evidence=evidence,
                current=InferenceTier.STANDARD,
                previous=InferenceTier.PRIORITY,
                reason=InferencePriorityTransitionReason.RECOVERY,
            )

        return None

    def __elevation_reason(self) -> Optional[InferencePriorityTransitionReason]:
        """
        Return why recent signals require priority.
        """

        evidence = self.__evidence()

        if evidence.failures >= evidence.threshold.failures:
            return InferencePriorityTransitionReason.TRANSIENT

        if evidence.slows >= evidence.threshold.slows:
            return InferencePriorityTransitionReason.SLOW

        return None

    def __evidence(self) -> PriorityInferenceEvidence:
        """
        Return bounded signal counts and thresholds for diagnostics.
        """

        failures = 0
        slows = 0

        for signal in self.__signals:
            if signal.transient:
                failures += 1

            if self.__is_slow(signal=signal):
                slows += 1

        return PriorityInferenceEvidence(
            slows=slows,
            failures=failures,
            healthy=self.__healthy,
            window=len(self.__signals),
            threshold=self.__configuration.adaptive.threshold,
        )

    def __is_healthy_priority(self, *, signal: PriorityInferenceSignal) -> bool:
        """
        Return whether a priority attempt is healthy enough to count toward recovery.
        """

        return (
            signal.success
            and not signal.transient
            and not self.__is_slow(signal=signal)
            and signal.tier == InferenceTier.PRIORITY
        )

    def __is_slow(self, *, signal: PriorityInferenceSignal) -> bool:
        """
        Return whether a successful attempt crossed the adaptive latency threshold.
        """

        if not signal.success or signal.latency is None:
            return False

        return signal.latency >= self.__configuration.adaptive.threshold.latency
