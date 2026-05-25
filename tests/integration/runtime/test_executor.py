from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

from fathom.adapters.signal.temporal import TemporalSignalAdapter
from fathom.constants import SignalType
from fathom.core.services.hitl import HITLService
from fathom.infrastructure.temporal.state import SignalStateRegistry
from fathom.runtime.executor import GraphExecutor


class TestGraphExecutorIntegration:
    """
    Covers executor behavior across runtime adapters.
    """

    @pytest.mark.asyncio
    async def test_temporal_cancel_signal_stops_paused_executor_stream(self) -> None:
        """
        A cancellation signal must wake pause waiting and cancel the graph stream.
        """

        workflow_id = f"integration-{uuid4()}"
        adapter = TemporalSignalAdapter(workflow_id=workflow_id)
        state = SignalStateRegistry.shared().get(workflow_id=workflow_id)
        telemetry = SimpleNamespace(info=AsyncMock())
        context = SimpleNamespace(
            is_cancelled=False,
            cancel=Mock(side_effect=lambda: setattr(context, "is_cancelled", True)),
            hitl=HITLService(signal=adapter, telemetry=telemetry),
            telemetry=telemetry,
        )
        executor = GraphExecutor(
            thread_id="thread-integration",
            context=context,  # type: ignore[arg-type]
            graph=SimpleNamespace(),  # type: ignore[arg-type]
        )
        stream_task = asyncio.create_task(asyncio.sleep(60))
        pause_waiter = asyncio.create_task(adapter.wait_for_pause())

        try:
            await asyncio.sleep(0)
            state.mark_cancelled()
            await asyncio.wait_for(pause_waiter, timeout=1)

            should_continue = await executor._GraphExecutor__handle_pause(  # type: ignore[attr-defined]
                stream_task=stream_task
            )

            assert should_continue is False
            assert context.is_cancelled is True
            assert stream_task.cancelled()
            assert await adapter.check_signal() == SignalType.CANCELLED.value
            telemetry.info.assert_awaited_once()
        finally:
            stream_task.cancel()
            SignalStateRegistry.shared().release(workflow_id=workflow_id)
