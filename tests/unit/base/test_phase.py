"""
Pins for the bounded and abandonable phase primitives used by post-terminal finalization.
"""

from __future__ import annotations

import asyncio
import time
import unittest

from fathom.base.phase import AbandonablePhase, BoundedPhase
from fathom.constants.finalization import FinalizationPhase
from fathom.core.exceptions import FinalizationTimeoutError


class BoundedPhaseTest(unittest.IsolatedAsyncioTestCase):
    """
    BoundedPhase must return the inner result on success and raise FinalizationTimeoutError on overrun.
    """

    async def test_returns_inner_result_on_success(self) -> None:
        """
        A fast awaitable inside the deadline returns its value to the caller.
        """

        async def __work() -> str:
            return "ok"

        phase = BoundedPhase(
            phase=FinalizationPhase.HISTORY_FLUSH,
            timeout=1.0,
            workflow_id="workflow-test",
        )

        result = await phase.execute(awaitable=__work())

        self.assertEqual(result, "ok")

    async def test_raises_finalization_timeout_error_on_overrun(self) -> None:
        """
        A slow awaitable that exceeds the deadline raises FinalizationTimeoutError with phase and workflow context.
        """

        async def __work() -> None:
            await asyncio.sleep(1.0)

        phase = BoundedPhase(
            phase=FinalizationPhase.HISTORY_FLUSH,
            timeout=0.05,
            workflow_id="workflow-test",
        )

        with self.assertRaises(FinalizationTimeoutError) as captured:
            await phase.execute(awaitable=__work())

        self.assertEqual(captured.exception.phase, FinalizationPhase.HISTORY_FLUSH.value)
        self.assertEqual(captured.exception.workflow_id, "workflow-test")

    async def test_propagates_inner_exception_without_wrapping(self) -> None:
        """
        Exceptions other than TimeoutError propagate out of execute() unchanged.
        """

        async def __work() -> None:
            raise ValueError("inner failure")

        phase = BoundedPhase(
            phase=FinalizationPhase.HISTORY_FLUSH,
            timeout=1.0,
        )

        with self.assertRaises(ValueError):
            await phase.execute(awaitable=__work())


class AbandonablePhaseTest(unittest.IsolatedAsyncioTestCase):
    """
    AbandonablePhase must return promptly on timeout even against cancellation-resistant awaitables.
    """

    async def test_returns_inner_result_on_success(self) -> None:
        """
        A fast awaitable inside the deadline returns its value to the caller.
        """

        async def __work() -> int:
            return 42

        phase = AbandonablePhase(
            phase=FinalizationPhase.RUNNER_CLEANUP,
            timeout=1.0,
        )

        result = await phase.execute(awaitable=__work())

        self.assertEqual(result, 42)

    async def test_returns_none_when_awaitable_resists_cancellation(self) -> None:
        """
        A cancellation-resistant awaitable must NOT delay execute() past the deadline.
        """

        async def __stubborn() -> None:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                await asyncio.Event().wait()  # swallow cancel, hang forever

        phase = AbandonablePhase(
            phase=FinalizationPhase.RUNNER_CLEANUP,
            timeout=0.1,
            workflow_id="workflow-test",
        )
        started_at = time.perf_counter()
        result = await phase.execute(awaitable=__stubborn())
        elapsed = time.perf_counter() - started_at

        self.assertIsNone(result)
        self.assertLess(
            elapsed,
            0.5,
            "AbandonablePhase must return promptly even when the awaitable swallows CancelledError",
        )

    async def test_swallows_inner_exception_and_returns_none(self) -> None:
        """
        An exception in the inner awaitable is logged and turned into a None result; never propagated.
        """

        async def __work() -> None:
            raise RuntimeError("inner failure")

        phase = AbandonablePhase(
            phase=FinalizationPhase.RUNNER_CLEANUP,
            timeout=1.0,
        )

        result = await phase.execute(awaitable=__work())

        self.assertIsNone(result)
