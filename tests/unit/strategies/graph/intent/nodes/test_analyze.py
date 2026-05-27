from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any, Dict
from unittest.mock import AsyncMock, Mock

from fathom.constants.state import CommonStateKey, CompletionReason, IntentStateKey
from fathom.core.agent.state import AgentState
from fathom.schemas.capabilities import HITLCapability, RuntimeCapabilities
from fathom.schemas.screens import ScreenCapture
from fathom.strategies.graph.intent.nodes.analyze import AnalyzeNode


class _Persistence:
    def __init__(self) -> None:
        self.last: Dict[Any, Any] = {}

    def restore(self, *, state: Dict[Any, Any]) -> None:
        _ = state

    def persist(self, *, result: Dict[Any, Any]) -> None:
        self.last = dict(result)


class AnalyzeNodeFailureBoundaryTest(unittest.IsolatedAsyncioTestCase):
    """
    Covers ANALYZE failure boundaries.
    """

    async def test_planner_exception_terminates_instead_of_retrying_forever(self) -> None:
        """
        Deterministic planner failures must fail fast instead of returning
        ``SHOULD_RETRY=True`` and creating a graph loop.
        """

        agent_state = AgentState(
            intent="finish onboarding",
            capabilities=RuntimeCapabilities(hitl=HITLCapability(enabled=False)),
        )
        planner = Mock()
        planner.plan_step = AsyncMock(side_effect=ValueError("bad planner state"))
        provider = SimpleNamespace(
            is_cancelled=AsyncMock(return_value=False),
            persistence=_Persistence(),
            hitl=SimpleNamespace(prompt=AsyncMock()),
            context=SimpleNamespace(
                workflow_id="run-test",
                max_steps=10,
                agent_state=agent_state,
                context_manager=SimpleNamespace(get_user_guidance=Mock(return_value=[])),
                device=SimpleNamespace(get_dimensions=AsyncMock(return_value=(100, 200))),
                signal=SimpleNamespace(supports_interruption=Mock(return_value=False)),
                configuration=SimpleNamespace(intent=SimpleNamespace(prompt_user_if_stuck=False)),
                planner=planner,
                use_xml=True,
                reasoner=Mock(),
                metrics=SimpleNamespace(record=Mock(), record_tokens=Mock()),
                telemetry=SimpleNamespace(info=AsyncMock(), error=AsyncMock()),
            ),
        )
        node = AnalyzeNode(provider=provider)  # type: ignore[arg-type]

        result = await node.run(
            state={
                CommonStateKey.CAPTURE: ScreenCapture(
                    width=100,
                    height=200,
                    activity="app",
                    image=b"png",
                    timestamp=1,
                )
            }
        )

        self.assertTrue(result[CommonStateKey.IS_COMPLETE])
        self.assertFalse(result[IntentStateKey.SHOULD_RETRY])
        self.assertEqual(result[CommonStateKey.COMPLETION_REASON], CompletionReason.FAILED.value)
        self.assertIn("bad planner state", result[CommonStateKey.FAILURE_DIAGNOSTIC])
        self.assertEqual(
            provider.persistence.last[CommonStateKey.COMPLETION_REASON],
            CompletionReason.FAILED.value,
        )
