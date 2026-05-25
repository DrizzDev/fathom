from __future__ import annotations

from types import SimpleNamespace
from typing import Dict
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from fathom.constants import ActionType
from fathom.constants.graph import NodeName
from fathom.constants.state import CommonStateKey, CompletionReason, IntentStateKey
from fathom.core.agent.state import AgentState
from fathom.schemas.actions import Action
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

        agent_state = AgentState(intent="tap continue", max_steps=8)
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
        assert provider.persistence.persist.called
