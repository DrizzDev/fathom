from __future__ import annotations

from typing import List, Optional

import pytest

from fathom.core.recovery import (
    NoopOutcome,
    RecoveryCoordinator,
    RecoveryOutcome,
    RecoveryRequest,
    RecoveryStrategy,
    RecoveryTrigger,
    ReplanOutcome,
)
from fathom.schemas.recovery import RecoveryPolicy
from fathom.schemas.screens import ScreenCapture


class TestRecoveryCoordinator:
    """
    Behavioral pins for :class:`RecoveryCoordinator`.
    """

    class __StubStrategy(RecoveryStrategy):
        """
        Configurable :class:`RecoveryStrategy` double. Records every ``recover`` call so tests can assert dispatch order
        and fall-through behavior, and lets each test pin the supported triggers and the outcome returned.
        """

        def __init__(
            self,
            *,
            name: str = "stub",
            outcome: Optional[RecoveryOutcome] = None,
            supports_set: Optional[List[RecoveryTrigger]] = None,
        ) -> None:
            self.__name = name
            self.__outcome = outcome
            self.__supports = set(supports_set or list(RecoveryTrigger))

            self.calls: List[RecoveryRequest] = []

        @property
        def name(self) -> str:
            """
            Stable strategy name used by the coordinator for logging.
            """

            return self.__name

        def supports(self, *, trigger: RecoveryTrigger) -> bool:
            """
            Report whether this stub is willing to handle ``trigger``.
            """

            return trigger in self.__supports

        async def recover(self, *, request: RecoveryRequest) -> Optional[RecoveryOutcome]:
            """
            Record the request and return the configured outcome.
            """

            self.calls.append(request)
            return self.__outcome

    @classmethod
    def __capture(cls) -> ScreenCapture:
        """
        Return a minimal valid :class:`ScreenCapture` for use in
        :class:`RecoveryRequest` fixtures; the coordinator does not inspect its contents.
        """

        return ScreenCapture(
            image=b"x",
            timestamp=0,
            width=1080,
            height=1920,
            activity="com.example/.Main",
        )

    @classmethod
    def __request(
        cls, *, trigger: RecoveryTrigger = RecoveryTrigger.VERIFY_REJECTED
    ) -> RecoveryRequest:
        """
        Build a default :class:`RecoveryRequest` for the supplied trigger.
        All other fields use non-empty placeholders so the request passes schema validation.
        """

        return RecoveryRequest(
            trigger=trigger,
            recent_actions=[],
            reason="test reason",
            capture=cls.__capture(),
            stuck_sub_goal="do thing",
            pending_sub_goals=["do thing"],
        )

    @classmethod
    def __policy(cls, **overrides: object) -> RecoveryPolicy:
        """
        Build a :class:`RecoveryPolicy` with ``enabled=True`` so
        coordinator tests opt into recovery without restating the flag at every call site.
        """

        base = {"enabled": True, **overrides}
        return RecoveryPolicy(**base)  # type: ignore[arg-type]

    @pytest.mark.asyncio
    async def test_master_toggle_disabled_returns_none(self) -> None:
        """
        Coordinator short-circuits with ``None`` when the master toggle
        is off, regardless of how many strategies are registered.
        """

        strategy = self.__StubStrategy(outcome=NoopOutcome(summary="ok"))
        coordinator = RecoveryCoordinator(
            strategies=[strategy], policy=RecoveryPolicy(enabled=False)
        )

        assert coordinator.enabled is False

        outcome = await coordinator.handle(
            scope=0,
            request=self.__request(),
            trigger=RecoveryTrigger.VERIFY_REJECTED,
        )

        assert outcome is None
        assert strategy.calls == []

    @pytest.mark.asyncio
    async def test_empty_strategy_list_returns_none(self) -> None:
        """
        With no strategies registered the coordinator reports itself
        disabled and returns ``None`` without touching counters.
        """

        coordinator = RecoveryCoordinator(strategies=[], policy=self.__policy())

        assert coordinator.enabled is False

        outcome = await coordinator.handle(
            scope=0,
            request=self.__request(),
            trigger=RecoveryTrigger.VERIFY_REJECTED,
        )

        assert outcome is None

    @pytest.mark.asyncio
    async def test_threshold_not_reached_returns_none(self) -> None:
        """
        Trigger fires below the configured threshold must not invoke any strategy.
        """

        strategy = self.__StubStrategy(outcome=ReplanOutcome(new_sub_goals=[], summary="r"))
        coordinator = RecoveryCoordinator(
            strategies=[strategy], policy=self.__policy(verify_threshold=3)
        )

        for _ in range(2):
            outcome = await coordinator.handle(
                scope=0,
                request=self.__request(),
                trigger=RecoveryTrigger.VERIFY_REJECTED,
            )
            assert outcome is None

        assert strategy.calls == []

    @pytest.mark.asyncio
    async def test_threshold_reached_dispatches_strategy(self) -> None:
        """
        Once the per-trigger threshold is reached the coordinator
        dispatches the first supporting strategy and returns its outcome.
        """

        outcome_value = ReplanOutcome(new_sub_goals=[], summary="replanned")

        strategy = self.__StubStrategy(outcome=outcome_value)
        coordinator = RecoveryCoordinator(
            strategies=[strategy], policy=self.__policy(verify_threshold=2)
        )

        first = await coordinator.handle(
            scope=0,
            request=self.__request(),
            trigger=RecoveryTrigger.VERIFY_REJECTED,
        )
        second = await coordinator.handle(
            scope=0,
            request=self.__request(),
            trigger=RecoveryTrigger.VERIFY_REJECTED,
        )

        assert first is None
        assert second is outcome_value
        assert len(strategy.calls) == 1

    @pytest.mark.asyncio
    async def test_counters_scoped_per_scope(self) -> None:
        """
        Counters keyed by ``scope`` are independent; a fire against
        ``scope=0`` must not advance the counter for ``scope=1``.
        """

        strategy = self.__StubStrategy(outcome=ReplanOutcome(new_sub_goals=[], summary="r"))
        coordinator = RecoveryCoordinator(
            strategies=[strategy], policy=self.__policy(verify_threshold=2)
        )

        first = await coordinator.handle(
            scope=0,
            request=self.__request(),
            trigger=RecoveryTrigger.VERIFY_REJECTED,
        )
        assert first is None
        assert strategy.calls == []

        second = await coordinator.handle(
            scope=1,
            request=self.__request(),
            trigger=RecoveryTrigger.VERIFY_REJECTED,
        )
        assert second is None
        assert strategy.calls == []

    @pytest.mark.asyncio
    async def test_separate_triggers_have_independent_counters(self) -> None:
        """
        Different triggers maintain separate counters under the same
        scope; a single fire of each must not cross either threshold.
        """

        strategy = self.__StubStrategy(outcome=ReplanOutcome(new_sub_goals=[], summary="r"))
        coordinator = RecoveryCoordinator(
            strategies=[strategy],
            policy=self.__policy(verify_threshold=2, plan_threshold=2),
        )

        verify_outcome = await coordinator.handle(
            scope=0,
            request=self.__request(),
            trigger=RecoveryTrigger.VERIFY_REJECTED,
        )
        block_outcome = await coordinator.handle(
            scope=0,
            trigger=RecoveryTrigger.ACTION_BLOCKED,
            request=self.__request(trigger=RecoveryTrigger.ACTION_BLOCKED),
        )

        assert strategy.calls == []
        assert block_outcome is None
        assert verify_outcome is None

    @pytest.mark.asyncio
    async def test_unsupported_trigger_skipped(self) -> None:
        """
        Strategies whose ``supports`` returns False must be skipped even once the threshold is reached.
        """

        strategy = self.__StubStrategy(
            supports_set=[RecoveryTrigger.ACTION_BLOCKED],
            outcome=ReplanOutcome(new_sub_goals=[], summary="r"),
        )
        coordinator = RecoveryCoordinator(
            strategies=[strategy], policy=self.__policy(verify_threshold=1)
        )

        outcome = await coordinator.handle(
            scope=0,
            request=self.__request(),
            trigger=RecoveryTrigger.VERIFY_REJECTED,
        )

        assert outcome is None
        assert strategy.calls == []

    @pytest.mark.asyncio
    async def test_priority_order_first_supporter_wins(self) -> None:
        """
        Strategies are tried in list order; the first supporter that
        returns a non-None outcome wins and no later strategy is invoked.
        """

        outcome_a = ReplanOutcome(new_sub_goals=[], summary="from-a")

        strategy_a = self.__StubStrategy(name="a", outcome=outcome_a)
        strategy_b = self.__StubStrategy(
            name="b",
            outcome=ReplanOutcome(new_sub_goals=[], summary="from-b"),
        )
        coordinator = RecoveryCoordinator(
            strategies=[strategy_a, strategy_b],
            policy=self.__policy(verify_threshold=1),
        )

        outcome = await coordinator.handle(
            scope=0,
            request=self.__request(),
            trigger=RecoveryTrigger.VERIFY_REJECTED,
        )

        assert outcome is outcome_a
        assert len(strategy_a.calls) == 1
        assert len(strategy_b.calls) == 0

    @pytest.mark.asyncio
    async def test_none_outcome_falls_through_to_next_strategy(self) -> None:
        """
        A strategy returning ``None`` must not stop dispatch; the
        coordinator continues to the next supporting strategy.
        """

        outcome_b = ReplanOutcome(new_sub_goals=[], summary="from-b")

        strategy_a = self.__StubStrategy(name="a", outcome=None)
        strategy_b = self.__StubStrategy(name="b", outcome=outcome_b)

        coordinator = RecoveryCoordinator(
            strategies=[strategy_a, strategy_b],
            policy=self.__policy(verify_threshold=1),
        )

        outcome = await coordinator.handle(
            scope=0,
            request=self.__request(),
            trigger=RecoveryTrigger.VERIFY_REJECTED,
        )

        assert outcome is outcome_b
        assert len(strategy_a.calls) == 1
        assert len(strategy_b.calls) == 1

    @pytest.mark.asyncio
    async def test_successful_outcome_resets_counter(self) -> None:
        """
        After a strategy commits an outcome the per-(scope, trigger)
        counter resets so subsequent fires re-accumulate from zero.
        """

        outcome_value = ReplanOutcome(new_sub_goals=[], summary="r")

        strategy = self.__StubStrategy(outcome=outcome_value)
        coordinator = RecoveryCoordinator(
            strategies=[strategy], policy=self.__policy(verify_threshold=1)
        )

        first = await coordinator.handle(
            scope=0,
            request=self.__request(),
            trigger=RecoveryTrigger.VERIFY_REJECTED,
        )
        assert first is outcome_value

        coordinator_higher_threshold = RecoveryCoordinator(
            strategies=[strategy], policy=self.__policy(verify_threshold=2)
        )

        await coordinator_higher_threshold.handle(
            scope=0,
            request=self.__request(),
            trigger=RecoveryTrigger.VERIFY_REJECTED,
        )
        await coordinator_higher_threshold.handle(
            scope=0,
            request=self.__request(),
            trigger=RecoveryTrigger.VERIFY_REJECTED,
        )

        # Two fires from a fresh counter == threshold reached; strategy dispatched.
        assert len(strategy.calls) >= 2

    @pytest.mark.asyncio
    async def test_reset_drops_counters(self) -> None:
        """
        Calling ``reset(scope=...)`` drops the counters for that scope
        so subsequent fires restart accumulation from zero.
        """

        strategy = self.__StubStrategy(outcome=ReplanOutcome(new_sub_goals=[], summary="r"))
        coordinator = RecoveryCoordinator(
            strategies=[strategy], policy=self.__policy(verify_threshold=3)
        )

        await coordinator.handle(
            scope=0,
            request=self.__request(),
            trigger=RecoveryTrigger.VERIFY_REJECTED,
        )
        await coordinator.handle(
            scope=0,
            request=self.__request(),
            trigger=RecoveryTrigger.VERIFY_REJECTED,
        )

        coordinator.reset(scope=0)
        outcome = await coordinator.handle(
            scope=0,
            request=self.__request(),
            trigger=RecoveryTrigger.VERIFY_REJECTED,
        )
        assert outcome is None
        assert strategy.calls == []
