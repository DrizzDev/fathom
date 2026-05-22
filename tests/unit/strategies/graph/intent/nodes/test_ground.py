from __future__ import annotations

import unittest
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from fathom.constants.state import CommonStateKey, CompletionReason
from fathom.strategies.graph.intent.nodes.ground import GroundNode


class GroundNodeEarlyExitTest(unittest.IsolatedAsyncioTestCase):
    """
    Pins the GROUND node's three early-exit branches.

    GROUND is the entry point of the LangGraph cycle. Three conditions
    must terminate the run before the perception port is even consulted:
    cancellation, max-step cap, and an empty screenshot from the device.
    Each branch must call ``agent_state.mark_complete`` with the right
    :class:`CompletionReason` and return an ``IS_COMPLETE=True`` patch so
    the graph short-circuits without producing a downstream ANALYZE call.
    """

    @staticmethod
    def __provider(
        *,
        cancelled: bool = False,
        step_count: int = 0,
        max_steps: int = 20,
        image: bytes = b"PNG",
        width: int = 1000,
        height: int = 2000,
    ) -> MagicMock:
        """
        Mocked :class:`IntentNodeProvider` exposing only the surface area
        GROUND touches on the early-exit branches.

        ``is_cancelled`` is an :class:`AsyncMock` because the node awaits
        it; ``perceive`` is mocked so the empty-screenshot path can be
        forced by passing ``image=b""``.
        """

        provider = MagicMock(name="IntentNodeProvider")
        provider.is_cancelled = AsyncMock(return_value=cancelled)
        provider.context.workflow_id = "run-test"
        provider.context.agent_state.step_count = step_count
        provider.context.max_steps = max_steps
        provider.context.telemetry.info = AsyncMock()
        provider.context.telemetry.error = AsyncMock()
        provider.context.perception.perceive = AsyncMock(
            return_value=MagicMock(image=image, width=width, height=height, activity="app"),
        )
        provider.persistence.persist = MagicMock()
        return provider

    async def test_cancellation_terminates_with_cancelled_reason(self) -> None:
        """
        A cancelled run must mark complete with :attr:`CompletionReason.CANCELLED`
        and never reach the perception port.
        """

        provider = self.__provider(cancelled=True)
        node = GroundNode(provider=provider)

        result: Any = await node(state={})  # type: ignore[arg-type]

        self.assertTrue(result.get(CommonStateKey.IS_COMPLETE))
        self.assertEqual(
            result.get(CommonStateKey.COMPLETION_REASON),
            CompletionReason.CANCELLED.value,
        )
        provider.context.agent_state.mark_complete.assert_called_once_with(
            reason=CompletionReason.CANCELLED.value,
        )

    async def test_step_count_at_cap_terminates_with_max_steps_reason(self) -> None:
        """
        Reaching the configured step cap before planning the next action
        must terminate with :attr:`CompletionReason.MAX_STEPS`, not
        FAILED. The cap is checked before any work to avoid spending a
        capture on a step that cannot execute.
        """

        provider = self.__provider(cancelled=False, step_count=20, max_steps=20)
        node = GroundNode(provider=provider)

        result: Any = await node(state={})  # type: ignore[arg-type]

        self.assertTrue(result.get(CommonStateKey.IS_COMPLETE))
        self.assertEqual(
            result.get(CommonStateKey.COMPLETION_REASON),
            CompletionReason.MAX_STEPS.value,
        )
        provider.context.perception.perceive.assert_not_awaited()

    async def test_empty_capture_terminates_with_failed_reason(self) -> None:
        """
        An empty screenshot from the perception port is a hard failure
        (device disconnected, lost surface, etc.) and must terminate with
        :attr:`CompletionReason.FAILED` so the run is surfaced as broken
        rather than silently looping on empty captures.
        """

        provider = self.__provider(cancelled=False, image=b"")
        node = GroundNode(provider=provider)

        result: Any = await node(state={})  # type: ignore[arg-type]

        self.assertTrue(result.get(CommonStateKey.IS_COMPLETE))
        self.assertEqual(
            result.get(CommonStateKey.COMPLETION_REASON),
            CompletionReason.FAILED.value,
        )
