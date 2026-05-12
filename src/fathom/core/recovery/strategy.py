from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from fathom.core.recovery.types import RecoveryOutcome, RecoveryRequest, RecoveryTrigger


class RecoveryStrategy(ABC):
    """
    Abstract base for a pluggable recovery strategy.

    Implementations are stateless across runs and receive the ports they
    need via constructor injection at the composition root. The coordinator
    queries ``supports`` to decide eligibility per trigger and dispatches ``recover`` once the per-trigger threshold has been reached.

    Returning ``None`` (or a ``NoopOutcome``) signals "I cannot help with this situation," and the coordinator continues to the next strategy.
    Returning a concrete outcome (e.g. ``ReplanOutcome``) commits the recovery decision and resets the coordinator's counters for that sub-goal.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Stable identifier used for configuration and logging.
        """

        raise NotImplementedError

    @abstractmethod
    def supports(self, *, trigger: RecoveryTrigger) -> bool:
        """
        Return whether this strategy is willing to handle the given trigger.
        """

        raise NotImplementedError

    @abstractmethod
    async def recover(self, *, request: RecoveryRequest) -> Optional[RecoveryOutcome]:
        """
        Attempt to recover from the stuck state described in ``request``.
        Returns the outcome on success or ``None`` to defer to later strategies.
        """

        raise NotImplementedError
