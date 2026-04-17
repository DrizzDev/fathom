from __future__ import annotations

import asyncio
import concurrent.futures
import unittest
from typing import List, Tuple
from unittest.mock import MagicMock

from fathom.adapters.signal.tui import TuiSignalAdapter


class _FakeApp:
    """
    Minimal DemoApp stub that records ``call_from_thread`` invocations
    and lets tests immediately resolve the HITL future, simulating
    what the Textual modal would do after a user submits a response.
    """

    def __init__(self) -> None:
        self.calls: List[Tuple[str, tuple, dict]] = []
        self.__canned_response: str = ""

    def set_canned_response(self, text: str) -> None:
        """Resolve the next HITL ask future to this text."""

        self.__canned_response = text

    def call_from_thread(self, callback, *args, **kwargs):
        """Resolve any ``concurrent.futures.Future`` arg synchronously."""

        self.calls.append((getattr(callback, "__name__", str(callback)), args, kwargs))
        for arg in list(args) + list(kwargs.values()):
            if isinstance(arg, concurrent.futures.Future):
                arg.set_result(self.__canned_response)

    # DemoApp.push_hitl_ask — exists so the adapter's callable reference
    # resolves cleanly when we don't want to go through the real app.
    def push_hitl_ask(
        self,
        prompt: str,
        future: "concurrent.futures.Future[str]",
    ) -> None:  # pragma: no cover - replaced by call_from_thread stub
        raise AssertionError("push_hitl_ask should only be invoked via call_from_thread in tests")


class TuiSignalAdapterStaticsTest(unittest.TestCase):
    """
    Cover the synchronous shape of TuiSignalAdapter.
    """

    def test_is_interactive_reports_true(self) -> None:
        adapter = TuiSignalAdapter(app=_FakeApp())  # type: ignore[arg-type]
        self.assertTrue(adapter.is_interactive)

    def test_supports_interruption_is_false_to_disable_langgraph_interrupts(self) -> None:
        adapter = TuiSignalAdapter(app=_FakeApp())  # type: ignore[arg-type]
        self.assertFalse(adapter.supports_interruption())


class TuiSignalAdapterAskTest(unittest.IsolatedAsyncioTestCase):
    """
    Cover the cross-thread future round-trip in ``ask``.
    """

    async def test_ask_returns_response_pushed_by_modal_stub(self) -> None:
        app = _FakeApp()
        app.set_canned_response("HSR Layout")
        adapter = TuiSignalAdapter(app=app)  # type: ignore[arg-type]

        answer = await adapter.ask(prompt="Where is the delivery address?")

        self.assertEqual(answer, "HSR Layout")
        # Exactly one modal was scheduled, passing the prompt as first arg.
        self.assertEqual(len(app.calls), 1)
        name, args, _ = app.calls[0]
        self.assertEqual(name, "push_hitl_ask")
        self.assertEqual(args[0], "Where is the delivery address?")
        self.assertIsInstance(args[1], concurrent.futures.Future)

    async def test_ask_raises_when_scheduling_fails(self) -> None:
        app = MagicMock()
        app.call_from_thread.side_effect = RuntimeError("no active app")
        adapter = TuiSignalAdapter(app=app)

        with self.assertRaises(RuntimeError):
            await adapter.ask(prompt="Anything?")

    async def test_ask_cancellation_cancels_pending_future(self) -> None:
        """
        Awaiting task cancellation must propagate and cancel the
        future so the main thread can abandon the modal cleanly.
        """

        class _SlowApp:
            """Never resolves the future — simulates an open modal."""

            def call_from_thread(self, _callback, *args, **_kwargs):
                # Don't set_result; the future stays pending so we can cancel the await.
                pass

            def push_hitl_ask(
                self,
                _prompt: str,
                _future: "concurrent.futures.Future[str]",
            ) -> None:  # pragma: no cover - reference-only stub
                return None

        adapter = TuiSignalAdapter(app=_SlowApp())  # type: ignore[arg-type]

        task = asyncio.create_task(adapter.ask(prompt="Will be cancelled"))
        # Yield once so the adapter schedules the modal and parks on the future.
        await asyncio.sleep(0)
        task.cancel()

        with self.assertRaises(asyncio.CancelledError):
            await task


class TuiSignalAdapterContextQueueTest(unittest.IsolatedAsyncioTestCase):
    """
    Cover the injected-context queue surface (FIFO, peek, consume).
    """

    async def test_inject_and_pop_round_trip(self) -> None:
        adapter = TuiSignalAdapter(app=_FakeApp())  # type: ignore[arg-type]
        adapter.inject_context("hint 1")
        adapter.inject_context("hint 2")

        self.assertTrue(await adapter.has_injected_context())
        self.assertEqual(await adapter.peek_next_context(), "hint 1")
        self.assertEqual(await adapter.get_injected_context(), "hint 1")
        self.assertEqual(await adapter.peek_next_context(), "hint 2")

    async def test_consume_without_get_advances_queue(self) -> None:
        adapter = TuiSignalAdapter(app=_FakeApp())  # type: ignore[arg-type]
        adapter.inject_context("first")
        adapter.inject_context("second")

        await adapter.consume_context()

        self.assertEqual(await adapter.peek_next_context(), "second")

    async def test_empty_queue_safe_operations(self) -> None:
        adapter = TuiSignalAdapter(app=_FakeApp())  # type: ignore[arg-type]

        self.assertFalse(await adapter.has_injected_context())
        self.assertIsNone(await adapter.peek_next_context())
        self.assertIsNone(await adapter.get_injected_context())
        # consume_context on empty queue should be a no-op.
        await adapter.consume_context()

    async def test_inject_ignores_empty_strings(self) -> None:
        adapter = TuiSignalAdapter(app=_FakeApp())  # type: ignore[arg-type]
        adapter.inject_context("")
        self.assertFalse(await adapter.has_injected_context())


class TuiSignalAdapterBlockingMethodsTest(unittest.IsolatedAsyncioTestCase):
    """
    Cover the methods that block or return trivial values.
    """

    async def test_check_signal_always_none(self) -> None:
        adapter = TuiSignalAdapter(app=_FakeApp())  # type: ignore[arg-type]
        self.assertIsNone(await adapter.check_signal())

    async def test_is_pause_requested_is_false(self) -> None:
        adapter = TuiSignalAdapter(app=_FakeApp())  # type: ignore[arg-type]
        self.assertFalse(await adapter.is_pause_requested())

    async def test_wait_for_resume_returns_immediately(self) -> None:
        adapter = TuiSignalAdapter(app=_FakeApp())  # type: ignore[arg-type]
        await asyncio.wait_for(adapter.wait_for_resume(), timeout=0.1)

    async def test_wait_for_pause_raises_not_implemented(self) -> None:
        adapter = TuiSignalAdapter(app=_FakeApp())  # type: ignore[arg-type]

        with self.assertRaises(NotImplementedError):
            await adapter.wait_for_pause()
