from __future__ import annotations

import asyncio
import unittest
from uuid import uuid4

from fathom.adapters.signal.temporal import TemporalSignalAdapter
from fathom.infrastructure.temporal.state import SignalStateRegistry


class TemporalSignalAdapterTest(unittest.IsolatedAsyncioTestCase):
    """
    Covers workflow signal waiting semantics.
    """

    async def asyncSetUp(self) -> None:
        self.workflow_id = f"test-{uuid4()}"

    async def asyncTearDown(self) -> None:
        SignalStateRegistry.shared().release(workflow_id=self.workflow_id)

    async def test_wait_for_pause_resolves_when_workflow_is_cancelled(self) -> None:
        """
        A cancellation signal must wake callers waiting for a pause signal.
        """

        adapter = TemporalSignalAdapter(workflow_id=self.workflow_id)
        state = SignalStateRegistry.shared().get(workflow_id=self.workflow_id)

        waiter = asyncio.create_task(adapter.wait_for_pause())
        await asyncio.sleep(0)

        state.mark_cancelled()

        await asyncio.wait_for(waiter, timeout=1)
        self.assertTrue(state.cancelled)
