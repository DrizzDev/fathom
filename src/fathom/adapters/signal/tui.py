"""
Signal adapter that routes HITL prompts through a Textual modal
instead of the stdin-driven ``InteractiveSignal`` adapter.

The demo TUI owns the terminal in raw mode, so the stdin reader that
``InteractiveSignal`` attaches via ``loop.add_reader`` can never fire —
any HITL call would deadlock the agent's worker thread.

This adapter bridges the two threads:

1. The agent's worker thread (its own asyncio loop) awaits a
   ``concurrent.futures.Future`` wrapped with ``asyncio.wrap_future``.
2. ``call_from_thread`` schedules a Textual ``ModalScreen`` push on
   the main thread.
3. When the user submits the modal, the screen calls
   ``future.set_result(...)`` — ``concurrent.futures.Future`` is
   thread-safe, so the agent thread observes completion and returns.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
from collections import deque
from logging import getLogger
from typing import TYPE_CHECKING, Optional

from fathom.interfaces.signal import SignalPort

if TYPE_CHECKING:
    from fathom.runtime.command.demo_tui import DemoApp

logger = getLogger(__name__)


class TuiSignalAdapter(SignalPort):
    """
    Cross-thread signal adapter backed by a Textual ``DemoApp``.

    ``supports_interruption()`` deliberately returns ``False`` so
    LangGraph does not configure ``interrupt_before`` between-step
    pauses — those require a blocking pause channel we don't provide.
    HITL is driven by the agent calling ``ask()`` directly when it
    needs a human decision.
    """

    def __init__(self, *, app: "DemoApp") -> None:
        """
        Bind the adapter to a ``DemoApp`` instance used to push modals.
        """

        self.__app = app
        self.__injected_contexts: deque[str] = deque()

    @property
    def is_interactive(self) -> bool:
        """
        Report interactive capability so downstream services treat the
        adapter as HITL-capable.
        """

        return True

    def supports_interruption(self) -> bool:
        """
        Disable LangGraph between-step interrupts.

        Enabling interrupts without a pause channel would block the
        agent waiting for a ``wait_for_pause`` signal that can never
        arrive through the TUI today.
        """

        return False

    async def check_signal(self) -> Optional[str]:
        """
        No background signals arrive in the TUI path; always ``None``.
        """

        return None

    async def wait_for_pause(self) -> None:
        """
        Block forever — should never be called while
        ``supports_interruption()`` is ``False``.
        """

        await asyncio.Event().wait()

    async def wait_for_resume(self) -> None:
        """
        No pause state to resume from in the TUI path.
        """

        return

    async def ask(self, *, prompt: str) -> str:
        """
        Show an HITL modal and block until the user submits a response.

        Implementation marshals across two asyncio loops:

        - Agent worker-thread loop: awaits ``asyncio.wrap_future(...)``
          on a ``concurrent.futures.Future`` owned by this call.
        - Main Textual loop: ``call_from_thread`` pushes the modal,
          and ``_HitlAskScreen.on_input_submitted`` sets the future
          result when the user presses Enter.

        Returns the raw string the user typed. Returns an empty string
        on modal scheduling failure so the agent can treat the call
        as a graceful no-response fallback.
        """

        future: concurrent.futures.Future[str] = concurrent.futures.Future()

        try:
            self.__app.call_from_thread(self.__app.push_hitl_ask, prompt, future)
        except Exception as exception:
            logger.warning("Failed to schedule HITL modal: %s", exception)
            return ""

        try:
            return await asyncio.wrap_future(future)
        except asyncio.CancelledError:
            # Agent-side cancellation (e.g. Ctrl-C) — signal the main
            # thread to drop any pending modal.
            future.cancel()
            raise

    async def get_injected_context(self) -> Optional[str]:
        """
        Pop the next injected context from the queue, or ``None``.
        """

        if self.__injected_contexts:
            return self.__injected_contexts.popleft()
        return None

    async def peek_next_context(self) -> Optional[str]:
        """
        Peek at the next injected context without consuming it.
        """

        return self.__injected_contexts[0] if self.__injected_contexts else None

    async def consume_context(self) -> None:
        """
        Drop the next context from the queue if one exists.
        """

        if self.__injected_contexts:
            self.__injected_contexts.popleft()

    async def is_pause_requested(self) -> bool:
        """
        Pause semantics not supported in the TUI path.
        """

        return False

    async def has_injected_context(self) -> bool:
        """
        Whether at least one injected context is queued.
        """

        return bool(self.__injected_contexts)

    def inject_context(self, text: str) -> None:
        """
        Push a context string into the queue.

        Callable from the main Textual thread after a future UI
        control (e.g. a "send context" keybinding) collects user
        input. Not used by the default HITL flow yet; present so the
        adapter is ready for richer TUI interactions later.
        """

        if text:
            self.__injected_contexts.append(text)
