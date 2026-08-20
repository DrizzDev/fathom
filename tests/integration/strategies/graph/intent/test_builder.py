from __future__ import annotations

from types import SimpleNamespace
from typing import Dict
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver

from fathom.constants import ActionType
from fathom.constants.graph import NodeName
from fathom.constants.state import CommonStateKey, CompletionReason, IntentStateKey
from fathom.core.agent.state import AgentState
from fathom.core.capability.catalog import CommandCatalogProvider
from fathom.core.exceptions import HITLNotAvailableError
from fathom.core.services.timing import RunClock
from fathom.schemas.actions import Action
from fathom.schemas.capabilities import HITLCapability, RuntimeCapabilities
from fathom.schemas.execution import ExecutionContext
from fathom.schemas.localization import LocalizationResult, LocalizationStatus
from fathom.schemas.screens import ScreenCapture
from fathom.schemas.steps import Step
from fathom.strategies.graph.intent.builder import IntentGraphBuilder
from fathom.strategies.graph.intent.nodes.execute import ExecuteNode
from fathom.strategies.graph.intent.nodes.factory import IntentGraphFactory
from fathom.strategies.graph.intent.nodes.observe import ObserveNode
from fathom.strategies.graph.intent.nodes.record import RecordNode
from fathom.strategies.graph.state import IntentGraphState


class TestIntentExecutionFlowIntegration:
    """
    Covers compiled intent-graph behavior across node boundaries.
    """

    @pytest.mark.asyncio
    async def test_compiled_graph_terminates_when_execution_context_is_missing(self) -> None:
        """
        A broken upstream node must fail the compiled graph instead of looping.
        """

        agent_state = AgentState(
            intent="tap continue",
            capabilities=RuntimeCapabilities(hitl=HITLCapability(enabled=False)),
            max_steps=8,
        )
        context = SimpleNamespace(
            agent_state=agent_state,
            is_cancelled=False,
            max_steps=8,
            clock=RunClock(),
            workflow_id="integration-graph-missing-context",
        )
        provider = MagicMock(name="IntentNodeProvider")
        provider.context = context
        provider.is_cancelled = AsyncMock(return_value=False)
        provider.persistence.persist = Mock()

        ground_calls = 0

        async def ground(_state: IntentGraphState) -> Dict[str, object]:
            nonlocal ground_calls
            ground_calls += 1
            return {}

        async def analyze(_state: IntentGraphState) -> Dict[str, object]:
            return {
                CommonStateKey.IS_COMPLETE: False,
                IntentStateKey.SHOULD_RETRY: False,
                IntentStateKey.PLANNED_STEP: Step(
                    action=Action(
                        action_type=ActionType.TAP,
                        rationale="continue onboarding",
                        target="Continue",
                        confidence=0.9,
                    ),
                    screen_hash="screen-before",
                    step_number=1,
                ),
            }

        async def broken_supervise(_state: IntentGraphState) -> Dict[str, object]:
            return {}

        nodes = {
            NodeName.GROUND: ground,
            NodeName.ANALYZE: analyze,
            NodeName.SUPERVISE: broken_supervise,
            NodeName.EXECUTE: ExecuteNode(provider=provider),
            NodeName.OBSERVE: ObserveNode(provider=provider),
            NodeName.RECORD: RecordNode(provider=provider),
            NodeName.VERIFY: broken_supervise,
        }

        with patch.object(IntentGraphFactory, "build", return_value=nodes):
            graph = IntentGraphBuilder(context=context).build()  # type: ignore[arg-type]
            result = await graph.ainvoke({}, config={"recursion_limit": 8})

        assert ground_calls == 1
        assert agent_state.is_complete
        assert result[CommonStateKey.IS_COMPLETE] is True
        assert result[CommonStateKey.COMPLETION_REASON] == CompletionReason.FAILED.value
        assert result[IntentStateKey.SHOULD_RETRY] is False


class TestHitlUnavailableReplan:
    """
    End-to-end pin for stale-ASK_USER recovery via the compiled graph.
    """

    @staticmethod
    def __capture() -> ScreenCapture:
        """
        Build a minimal screen capture.
        """

        return ScreenCapture(width=100, height=200, activity="app", image=b"", timestamp=1)

    @staticmethod
    def __ask_user_context() -> ExecutionContext:
        """
        Execution context whose planned step is ASK_USER (stale replay).
        """

        action = Action(
            action_type=ActionType.ASK_USER,
            target="User",
            confidence=1.0,
            rationale="missing credentials",
            text="What is your OTP?",
        )
        step = Step(step_number=3, screen_hash="v", action=action)
        return ExecutionContext(
            step=step,
            capture=TestHitlUnavailableReplan.__capture(),
            localization=LocalizationResult(
                status=LocalizationStatus.UNRESOLVED,
                bounds=None,
                source=None,
                confidence=0.0,
                reason="ask_user_bypass",
            ),
            package="app",
        )

    @pytest.mark.asyncio
    async def test_stale_ask_user_routes_execute_back_to_ground(self) -> None:
        """
        A replayed ASK_USER step under autonomous capabilities must reach GROUND, not OBSERVE.
        """

        agent_state = AgentState(
            intent="login",
            capabilities=RuntimeCapabilities(hitl=HITLCapability(enabled=False)),
            max_steps=5,
        )
        context = SimpleNamespace(
            agent_state=agent_state,
            is_cancelled=False,
            max_steps=5,
            catalog=CommandCatalogProvider().build(),
            clock=RunClock(),
            workflow_id="integration-hitl-unavailable",
        )
        provider = MagicMock(name="IntentNodeProvider")
        provider.context = context
        provider.is_cancelled = AsyncMock(return_value=False)
        provider.persistence.persist = Mock()
        provider.hitl.available = Mock(return_value=False)
        provider.hitl.ask = AsyncMock(side_effect=HITLNotAvailableError())

        ground_calls = 0
        analyze_calls = 0
        observe_calls = 0

        async def ground(_state: IntentGraphState) -> Dict[str, object]:
            nonlocal ground_calls
            ground_calls += 1
            if ground_calls >= 2:
                agent_state.mark_complete(reason=CompletionReason.CANCELLED.value)
                return {
                    CommonStateKey.IS_COMPLETE: True,
                    CommonStateKey.COMPLETION_REASON: CompletionReason.CANCELLED.value,
                }
            return {}

        async def analyze(_state: IntentGraphState) -> Dict[str, object]:
            nonlocal analyze_calls
            analyze_calls += 1
            return {
                IntentStateKey.PLANNED_STEP: TestHitlUnavailableReplan.__ask_user_context().step,
                CommonStateKey.IS_COMPLETE: False,
                IntentStateKey.SHOULD_RETRY: False,
            }

        async def supervise(_state: IntentGraphState) -> Dict[str, object]:
            return {
                IntentStateKey.EXECUTION_CONTEXT: TestHitlUnavailableReplan.__ask_user_context()
            }

        async def observe(_state: IntentGraphState) -> Dict[str, object]:
            nonlocal observe_calls
            observe_calls += 1
            return {}

        async def passthrough(_state: IntentGraphState) -> Dict[str, object]:
            return {}

        nodes = {
            NodeName.GROUND: ground,
            NodeName.ANALYZE: analyze,
            NodeName.SUPERVISE: supervise,
            NodeName.EXECUTE: ExecuteNode(provider=provider),
            NodeName.OBSERVE: observe,
            NodeName.RECORD: passthrough,
            NodeName.VERIFY: passthrough,
        }

        with patch.object(IntentGraphFactory, "build", return_value=nodes):
            graph = IntentGraphBuilder(context=context).build()  # type: ignore[arg-type]
            await graph.ainvoke({}, config={"recursion_limit": 12})

        assert observe_calls == 0
        assert ground_calls >= 2
        # HITL is unavailable, so EXECUTE replans up front without opening a persisted step or asking.
        provider.hitl.ask.assert_not_awaited()
        assert provider.persistence.persist.called


class TestGroundRoutingReadsReturnedState:
    """
    Pins that GROUND routing consumes the merged node-return patch rather than the
    mutable shared ``context.agent_state``.

    Each test builds the real compiled graph so the real ``__route_after_ground``
    conditional edge runs; only the node bodies are stubbed. ``context.agent_state``
    is deliberately left incomplete so a router that still read it would misroute,
    making these tests fail closed against a regression to the pre-Phase-1 source.
    """

    __TERMINAL = CompletionReason.STUCK.value

    @staticmethod
    def __config(*, thread_id: str) -> RunnableConfig:
        """
        Build a checkpointer-scoped run configuration for the given thread.
        """

        return {"recursion_limit": 8, "configurable": {"thread_id": thread_id}}

    @staticmethod
    def __context() -> SimpleNamespace:
        """
        Build a synthetic graph context whose agent_state never reports completion.
        """

        return SimpleNamespace(
            is_cancelled=False,
            agent_state=SimpleNamespace(is_complete=False, completion_reason=None),
            workflow_id="integration-ground-routing",
        )

    @classmethod
    def __nodes(
        cls, *, ground_patch: Dict[str, object], analyze_ran: list[bool]
    ) -> Dict[str, object]:
        """
        Build the seven node callables, recording whether ANALYZE was reached.
        """

        async def ground(_state: IntentGraphState) -> Dict[str, object]:
            return dict(ground_patch)

        async def analyze(_state: IntentGraphState) -> Dict[str, object]:
            analyze_ran.append(True)
            return {
                CommonStateKey.IS_COMPLETE: True,
                CommonStateKey.COMPLETION_REASON: cls.__TERMINAL,
            }

        async def passthrough(_state: IntentGraphState) -> Dict[str, object]:
            return {}

        return {
            NodeName.GROUND: ground,
            NodeName.ANALYZE: analyze,
            NodeName.SUPERVISE: passthrough,
            NodeName.EXECUTE: passthrough,
            NodeName.OBSERVE: passthrough,
            NodeName.RECORD: passthrough,
            NodeName.VERIFY: passthrough,
        }

    @pytest.mark.asyncio
    async def test_terminal_completion_in_returned_state_ends_run(self) -> None:
        """
        A terminal completion returned by GROUND ends the run without reaching ANALYZE.
        """

        analyze_ran: list[bool] = []
        context = self.__context()
        nodes = self.__nodes(
            ground_patch={
                CommonStateKey.IS_COMPLETE: True,
                CommonStateKey.COMPLETION_REASON: self.__TERMINAL,
            },
            analyze_ran=analyze_ran,
        )

        with patch.object(IntentGraphFactory, "build", return_value=nodes):
            graph = IntentGraphBuilder(context=context).build(checkpointer=MemorySaver())  # type: ignore[arg-type]
            result = await graph.ainvoke({}, config=self.__config(thread_id="terminal"))

        assert analyze_ran == []
        assert result[CommonStateKey.IS_COMPLETE] is True
        assert result[CommonStateKey.COMPLETION_REASON] == self.__TERMINAL

    @pytest.mark.asyncio
    async def test_happy_path_returned_state_advances_to_analyze(self) -> None:
        """
        A GROUND patch carrying no completion advances to ANALYZE.
        """

        analyze_ran: list[bool] = []
        context = self.__context()
        nodes = self.__nodes(
            ground_patch={IntentStateKey.SHOULD_RETRY: False},
            analyze_ran=analyze_ran,
        )

        with patch.object(IntentGraphFactory, "build", return_value=nodes):
            graph = IntentGraphBuilder(context=context).build(checkpointer=MemorySaver())  # type: ignore[arg-type]
            await graph.ainvoke({}, config=self.__config(thread_id="happy"))

        assert analyze_ran == [True]

    @pytest.mark.asyncio
    async def test_interrupt_before_ground_then_resume_routes_from_returned_state(self) -> None:
        """
        With execution resumed just before GROUND, LangGraph merges GROUND's returned
        patch before evaluating the edge, and routing ends the run without the mutable context.
        """

        analyze_ran: list[bool] = []
        context = self.__context()
        nodes = self.__nodes(
            ground_patch={
                CommonStateKey.IS_COMPLETE: True,
                CommonStateKey.COMPLETION_REASON: self.__TERMINAL,
            },
            analyze_ran=analyze_ran,
        )
        config = self.__config(thread_id="resume")

        with patch.object(IntentGraphFactory, "build", return_value=nodes):
            graph = IntentGraphBuilder(context=context).build(  # type: ignore[arg-type]
                checkpointer=MemorySaver(),
                interrupt_before=[NodeName.GROUND],
            )

            paused = await graph.ainvoke({}, config=config)
            snapshot = graph.get_state(config)

            assert paused is None
            assert snapshot.next == (NodeName.GROUND,)
            assert CommonStateKey.IS_COMPLETE not in snapshot.values
            assert analyze_ran == []

            resumed = await graph.ainvoke(None, config=config)

        assert analyze_ran == []
        assert resumed[CommonStateKey.IS_COMPLETE] is True
        assert resumed[CommonStateKey.COMPLETION_REASON] == self.__TERMINAL
