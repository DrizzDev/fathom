from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from tests.builders.agent import AgentFixtures

from fathom.constants import SignalType
from fathom.core.agent.state import AgentState
from fathom.runtime.executor import GraphExecutor
from fathom.schemas.abort import AbortDecision
from fathom.schemas.capabilities import HITLCapability, RuntimeCapabilities


class GraphExecutorRealignmentTest(unittest.IsolatedAsyncioTestCase):
    """
    Covers HITL context-injection state updates.
    """

    async def test_context_injection_records_one_canonical_realignment(self) -> None:
        """
        The executor must use AgentState's realignment tracker as the
        single source of truth instead of keeping a second counter.
        """

        agent_state = AgentState(
            intent="finish onboarding",
            capabilities=RuntimeCapabilities(hitl=HITLCapability(enabled=True)),
        )
        context = SimpleNamespace(
            agent_state=agent_state,
            context_manager=AgentFixtures.context_manager(),
        )
        graph = SimpleNamespace(aupdate_state=AsyncMock())
        executor = GraphExecutor(
            thread_id="thread-test",
            context=context,  # type: ignore[arg-type]
            graph=graph,  # type: ignore[arg-type]
        )

        await executor._GraphExecutor__inject_context("tap Continue")  # type: ignore[attr-defined]

        self.assertEqual(agent_state.runtime.realignment.count, 1)
        context.context_manager.inject_user_guidance.assert_awaited_once()
        graph.aupdate_state.assert_awaited_once()

    async def test_pause_handler_cancels_stream_when_signal_is_cancelled(self) -> None:
        """
        A cancel signal that wakes the pause waiter must stop the in-flight stream.
        """

        context = SimpleNamespace(
            is_cancelled=False,
            cancel=Mock(side_effect=lambda: setattr(context, "is_cancelled", True)),
            hitl=SimpleNamespace(check_signal=AsyncMock(return_value=SignalType.CANCELLED.value)),
            telemetry=SimpleNamespace(info=AsyncMock()),
        )
        executor = GraphExecutor(
            thread_id="thread-test",
            context=context,  # type: ignore[arg-type]
            graph=SimpleNamespace(),  # type: ignore[arg-type]
        )
        stream_task = asyncio.create_task(asyncio.sleep(60))

        should_continue = await executor._GraphExecutor__handle_pause(  # type: ignore[attr-defined]
            stream_task=stream_task
        )

        self.assertFalse(should_continue)
        self.assertTrue(context.is_cancelled)
        self.assertTrue(stream_task.cancelled())
        context.telemetry.info.assert_awaited_once()

    async def test_injected_abort_context_cancels_before_guidance_injection(self) -> None:
        """
        Manual injected stop text must be consumed and cancelled before planner guidance.
        """

        injected_contexts = ["end this test run"]

        async def has_injected_context() -> bool:
            return bool(injected_contexts)

        async def peek_next_context() -> str | None:
            return injected_contexts[0] if injected_contexts else None

        async def consume_context() -> None:
            if injected_contexts:
                injected_contexts.pop(0)

        context = SimpleNamespace(
            is_cancelled=False,
            cancel=Mock(side_effect=lambda: setattr(context, "is_cancelled", True)),
            abort_detector=SimpleNamespace(
                aborted=AsyncMock(
                    return_value=AbortDecision(aborted=True, confidence=0.99, fallback=False)
                )
            ),
            hitl=SimpleNamespace(
                check_signal=AsyncMock(return_value=SignalType.PAUSE.value),
                wait_for_resume=AsyncMock(),
                has_injected_context=AsyncMock(side_effect=has_injected_context),
                peek_next_context=AsyncMock(side_effect=peek_next_context),
                consume_context=AsyncMock(side_effect=consume_context),
            ),
            phase=SimpleNamespace(
                pause=AsyncMock(),
                resume=AsyncMock(),
                shutdown=AsyncMock(),
            ),
            telemetry=SimpleNamespace(info=AsyncMock()),
            agent_state=AgentState(
                intent="finish onboarding",
                capabilities=RuntimeCapabilities(hitl=HITLCapability(enabled=True)),
            ),
            context_manager=AgentFixtures.context_manager(),
        )
        graph = SimpleNamespace(aupdate_state=AsyncMock())
        executor = GraphExecutor(
            thread_id="thread-test",
            context=context,  # type: ignore[arg-type]
            graph=graph,  # type: ignore[arg-type]
        )

        await executor._GraphExecutor__handle_interrupt(source="manual_pause")  # type: ignore[attr-defined]

        self.assertTrue(context.is_cancelled)
        self.assertEqual(injected_contexts, [])
        context.abort_detector.aborted.assert_awaited_once()
        context.context_manager.inject_user_guidance.assert_not_awaited()
        graph.aupdate_state.assert_not_awaited()
