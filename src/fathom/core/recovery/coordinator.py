from __future__ import annotations

from logging import getLogger
from typing import Dict, List, Optional, Tuple

from fathom.core.recovery.strategy import RecoveryStrategy
from fathom.core.recovery.types import RecoveryOutcome, RecoveryRequest, RecoveryTrigger
from fathom.schemas.recovery import RecoveryPolicy

logger = getLogger(__name__)


class RecoveryCoordinator:
    """
    Dispatches recovery triggers to an ordered chain of strategies. Owns
    per-(scope, trigger) counters so escalation thresholds are tracked
    independently of agent state. ``scope`` is an opaque integer the
    caller uses to group counters (today: the active sub-goal index).

    Concurrency: instance-scoped per agent run. Designed for
    single-threaded asyncio access — counter read-modify-write is not
    atomic and is unsafe under threading. Document if the execution
    model changes.
    """

    def __init__(
        self,
        *,
        policy: RecoveryPolicy,
        strategies: List[RecoveryStrategy],
    ) -> None:
        self.__policy = policy
        self.__strategies = strategies
        self.__counters: Dict[Tuple[int, RecoveryTrigger], int] = {}

    @property
    def enabled(self) -> bool:
        """
        Whether recovery is operational (master switch on and strategies registered).
        """

        return self.__policy.enabled and bool(self.__strategies)

    @property
    def policy(self) -> RecoveryPolicy:
        """
        Active policy (read-only).
        """

        return self.__policy

    @property
    def strategy_names(self) -> List[str]:
        """
        Names of dispatch-eligible strategies in priority order.
        """

        return [strategy.name for strategy in self.__strategies]

    def reset(self, *, scope: int) -> None:
        """
        Drop counters for ``scope`` (e.g. on sub-goal advancement).
        Raises :class:`ValueError` for negative scope so caller bugs
        surface immediately rather than silently no-op.
        """

        if scope < 0:
            raise ValueError(f"scope must be non-negative, got {scope}")

        for key in [key for key in self.__counters if key[0] == scope]:
            del self.__counters[key]

    async def handle(
        self,
        *,
        scope: int,
        trigger: RecoveryTrigger,
        request: RecoveryRequest,
    ) -> Optional[RecoveryOutcome]:
        """
        Process a recovery trigger. Returns the first strategy outcome or
        ``None`` if disabled, below threshold, or no strategy handled it.
        """

        if not self.__policy.enabled:
            logger.info(
                "[RecoveryCoordinator] disabled via policy — skipping",
                extra={"component": "recovery", "event": "disabled", "trigger": trigger.value},
            )
            return None

        if not self.__strategies:
            logger.info(
                "[RecoveryCoordinator] no strategies registered — skipping",
                extra={"component": "recovery", "event": "no_strategies", "trigger": trigger.value},
            )
            return None

        key = (scope, trigger)
        count = self.__counters.get(key, 0) + 1

        self.__counters[key] = count
        threshold = self.__threshold_for(trigger=trigger)

        if count < threshold:
            logger.info(
                "[RecoveryCoordinator] %s below threshold (%d/%d) scope=%d",
                trigger.value,
                count,
                threshold,
                scope,
                extra={
                    "scope": scope,
                    "count": count,
                    "component": "recovery",
                    "threshold": threshold,
                    "trigger": trigger.value,
                    "event": "below_threshold",
                },
            )
            return None

        logger.info(
            "[RecoveryCoordinator] %s threshold reached (%d/%d) scope=%d — dispatching %s",
            trigger.value,
            count,
            threshold,
            scope,
            self.strategy_names,
            extra={
                "scope": scope,
                "count": count,
                "event": "dispatch",
                "component": "recovery",
                "threshold": threshold,
                "trigger": trigger.value,
                "strategies": self.strategy_names,
            },
        )

        for strategy in self.__strategies:
            if not strategy.supports(trigger=trigger):
                logger.info(
                    "[RecoveryCoordinator] strategy %r does not support %s",
                    strategy.name,
                    trigger.value,
                    extra={
                        "component": "recovery",
                        "trigger": trigger.value,
                        "strategy": strategy.name,
                        "event": "strategy_skipped",
                    },
                )
                continue

            if (outcome := await strategy.recover(request=request)) is None:
                logger.info(
                    "[RecoveryCoordinator] strategy %r declined",
                    strategy.name,
                    extra={
                        "scope": scope,
                        "component": "recovery",
                        "trigger": trigger.value,
                        "strategy": strategy.name,
                        "event": "strategy_declined",
                    },
                )
                continue

            logger.info(
                "[RecoveryCoordinator] %s committed: %s",
                strategy.name,
                outcome.summary,
                extra={
                    "scope": scope,
                    "component": "recovery",
                    "trigger": trigger.value,
                    "summary": outcome.summary,
                    "strategy": strategy.name,
                    "outcome_kind": outcome.kind,
                    "event": "strategy_committed",
                },
            )
            self.__counters[key] = 0
            return outcome

        logger.warning(
            "[RecoveryCoordinator] no strategy handled %s scope=%d",
            trigger.value,
            scope,
            extra={
                "scope": scope,
                "event": "no_handler",
                "component": "recovery",
                "trigger": trigger.value,
                "strategies": self.strategy_names,
            },
        )
        return None

    def __threshold_for(self, *, trigger: RecoveryTrigger) -> int:
        """
        Resolve the configured threshold for ``trigger``.
        """

        if trigger == RecoveryTrigger.VERIFY_REJECTED:
            return self.__policy.verify_threshold

        return self.__policy.plan_threshold
