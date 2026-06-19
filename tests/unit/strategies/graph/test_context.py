from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, MagicMock

from fathom.core.services.telemetry import PhaseAnnouncer
from fathom.strategies.graph.context import GraphContext


class _DrainableService:
    """
    Minimal stub exposing the drain_background_tasks surface GraphContext.shutdown calls on owned services.
    """

    def __init__(self) -> None:
        """
        Initialise the recorder that the GraphContext.shutdown drain pass triggers.
        """

        self.drain_called: AsyncMock = AsyncMock()

    async def drain_background_tasks(self) -> None:
        """
        Record that the drain ran even when phase.shutdown raised earlier in the teardown order.
        """

        await self.drain_called()


class GraphContextShutdownPhaseTest(unittest.IsolatedAsyncioTestCase):
    """
    Regression: GraphContext.shutdown must cancel the PhaseAnnouncer pulse so PHASE_HEARTBEAT events never outlive the workflow.
    The bound prod symptom is a leaked asyncio.Task that keeps firing 'Still working...' for minutes after WORKFLOW_COMPLETED.
    """

    @staticmethod
    def __build_context(
        *,
        phase: PhaseAnnouncer,
        action_executor: object = None,
    ) -> MagicMock:
        """
        Build a GraphContext spec-mock with the name-mangled private fields shutdown reads.
        """

        context = MagicMock(spec=GraphContext)

        context._GraphContext__phase = phase
        context._GraphContext__history = None
        context._GraphContext__hierarchy = None
        context._GraphContext__artifact_pipeline = None
        context._GraphContext__action_executor = action_executor

        return context

    async def test_shutdown_invokes_phase_shutdown(self) -> None:
        """
        Calling shutdown on a GraphContext must invoke phase.shutdown exactly once.
        """

        phase = MagicMock(spec=PhaseAnnouncer)

        phase.shutdown = AsyncMock()
        context = self.__build_context(phase=phase)

        await GraphContext.shutdown(context)

        phase.shutdown.assert_awaited_once()

    async def test_shutdown_continues_when_phase_shutdown_raises(self) -> None:
        """
        Phase-shutdown failures must be logged and swallowed so the rest of the teardown still runs.
        """

        phase = MagicMock(spec=PhaseAnnouncer)

        phase.shutdown = AsyncMock(side_effect=RuntimeError("simulated"))
        service = _DrainableService()

        context = self.__build_context(phase=phase, action_executor=service)

        await GraphContext.shutdown(context)

        phase.shutdown.assert_awaited_once()
        service.drain_called.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
