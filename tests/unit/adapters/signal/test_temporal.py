from __future__ import annotations

import asyncio
import unittest
from uuid import uuid4

from fathom.adapters.signal.temporal import TemporalSignalAdapter
from fathom.core.exceptions import HITLTimeoutError, WorkflowCancelledError
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

    async def test_wait_for_resume_raises_when_workflow_is_cancelled_during_pause(self) -> None:
        """
        Cancelling a paused workflow must surface WorkflowCancelledError to the executor.
        """

        adapter = TemporalSignalAdapter(workflow_id=self.workflow_id)
        state = SignalStateRegistry.shared().get(workflow_id=self.workflow_id)

        state.mark_paused()

        waiter = asyncio.create_task(adapter.wait_for_resume())
        await asyncio.sleep(0)

        state.mark_cancelled()

        with self.assertRaises(WorkflowCancelledError):
            await asyncio.wait_for(waiter, timeout=1)

    async def test_ask_raises_when_workflow_is_cancelled(self) -> None:
        """
        Cancelling an in-flight ask must surface WorkflowCancelledError so the graph unwinds.
        """

        adapter = TemporalSignalAdapter(workflow_id=self.workflow_id)
        state = SignalStateRegistry.shared().get(workflow_id=self.workflow_id)

        waiter = asyncio.create_task(adapter.ask(prompt="tap login"))
        await asyncio.sleep(0)

        state.mark_cancelled()

        with self.assertRaises(WorkflowCancelledError):
            await asyncio.wait_for(waiter, timeout=1)

    async def test_ask_raises_typed_timeout_at_deadline(self) -> None:
        """
        An unanswered ask must end in HITLTimeoutError at the deadline, never hang.
        """

        adapter = TemporalSignalAdapter(workflow_id=self.workflow_id, deadline=0.0)

        with self.assertRaises(HITLTimeoutError):
            await asyncio.wait_for(adapter.ask(prompt="Which account?"), timeout=1)
