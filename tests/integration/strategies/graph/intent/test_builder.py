from __future__ import annotations

from types import SimpleNamespace
from typing import Dict
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from fathom.constants import ActionType
from fathom.constants.graph import NodeName
from fathom.constants.state import CommonStateKey, CompletionReason, IntentStateKey
from fathom.core.agent.state import AgentState
from fathom.core.exceptions import HITLNotAvailableError
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
    """End-to-end pin for stale-ASK_USER recovery via the compiled graph."""

    @staticmethod
    def __capture() -> ScreenCapture:
        """Build a minimal screen capture."""

        return ScreenCapture(width=100, height=200, activity="app", image=b"", timestamp=1)

    @staticmethod
    def __ask_user_context() -> ExecutionContext:
        """Execution context whose planned step is ASK_USER (stale replay)."""

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
        """A replayed ASK_USER step under autonomous capabilities must reach GROUND, not OBSERVE."""

        agent_state = AgentState(
            intent="login",
            capabilities=RuntimeCapabilities(hitl=HITLCapability(enabled=False)),
            max_steps=5,
        )
        context = SimpleNamespace(
            agent_state=agent_state,
            is_cancelled=False,
            max_steps=5,
            workflow_id="integration-hitl-unavailable",
        )
        provider = MagicMock(name="IntentNodeProvider")
        provider.context = context
        provider.is_cancelled = AsyncMock(return_value=False)
        provider.persistence.persist = Mock()
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
        provider.hitl.ask.assert_awaited()
        assert provider.persistence.persist.called
