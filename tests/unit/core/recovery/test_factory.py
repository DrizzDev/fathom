from __future__ import annotations

from typing import Optional

from fathom.core.recovery import (
    RecoveryContext,
    RecoveryOutcome,
    RecoveryRequest,
    RecoveryStrategy,
    RecoveryStrategyFactory,
    RecoveryTrigger,
)


class TestRecoveryStrategyFactory:
    """
    Behavioral pins for :class:`RecoveryStrategyFactory`.
    """

    class __Custom(RecoveryStrategy):
        """
        Tagged :class:`RecoveryStrategy` double whose ``name`` exposes
        the tag so tests can assert that :meth:`build` returns instances in the expected order.
        """

        def __init__(self, *, tag: str) -> None:
            self.__tag = tag

        @property
        def name(self) -> str:
            """
            Return a tag-derived stable name for assertion purposes.
            """

            return f"custom:{self.__tag}"

        def supports(self, *, trigger: RecoveryTrigger) -> bool:
            """
            Accept every trigger; factory tests do not exercise dispatch.
            """

            _ = trigger
            return True

        async def recover(self, *, request: RecoveryRequest) -> Optional[RecoveryOutcome]:
            """
            Decline every request; factory tests do not invoke recovery.
            """

            _ = request
            return None

    def test_builtin_replan_registered(self) -> None:
        """
        The built-in ``replan`` strategy must be present in the factory without any explicit registration call.
        """

        assert "replan" in RecoveryStrategyFactory.available()

    def test_register_and_build_custom_strategy(self, llm_port_stub, memory_port_stub) -> None:
        """
        A custom strategy registered via :meth:`register` must be
        resolvable by name and constructed via the supplied builder.
        """

        RecoveryStrategyFactory.register("custom_test", lambda _ctx: self.__Custom(tag="x"))
        assert "custom_test" in RecoveryStrategyFactory.available()

        context = RecoveryContext(llm=llm_port_stub, memory=memory_port_stub)
        built = RecoveryStrategyFactory.build(names=["custom_test"], context=context)

        assert len(built) == 1
        assert built[0].name == "custom:x"

    def test_unknown_strategy_name_skipped(self, llm_port_stub, memory_port_stub) -> None:
        """
        Unknown strategy names must be skipped silently so a typo in
        ``RecoveryPolicy.strategies`` degrades gracefully.
        """

        context = RecoveryContext(llm=llm_port_stub, memory=memory_port_stub)
        built = RecoveryStrategyFactory.build(
            names=["definitely_not_real_strategy"], context=context
        )
        assert built == []

    def test_priority_order_preserved(self, llm_port_stub, memory_port_stub) -> None:
        """
        :meth:`build` must construct strategies in the order their names
        appear in the input list so coordinator priority is respected.
        """

        RecoveryStrategyFactory.register("a_order", lambda _ctx: self.__Custom(tag="a"))
        RecoveryStrategyFactory.register("b_order", lambda _ctx: self.__Custom(tag="b"))
        context = RecoveryContext(llm=llm_port_stub, memory=memory_port_stub)

        built_ab = RecoveryStrategyFactory.build(names=["a_order", "b_order"], context=context)
        built_ba = RecoveryStrategyFactory.build(names=["b_order", "a_order"], context=context)

        assert [strategy.name for strategy in built_ab] == ["custom:a", "custom:b"]
        assert [strategy.name for strategy in built_ba] == ["custom:b", "custom:a"]
