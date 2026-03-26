from __future__ import annotations

import asyncio
import contextlib
from logging import getLogger
from typing import Optional

from temporalio import activity

from fathom.constants import SIGNAL_HEARTBEAT_INTERVAL, SignalType
from fathom.infrastructure.temporal.state import SignalStateRegistry, WorkflowSignalState
from fathom.interfaces.signal import SignalPort

logger = getLogger(__name__)


class TemporalSignalAdapter(SignalPort):
    """
    Signal adapter backed by in-process shared state.

    Workflow signal handlers mirror state into a shared WorkflowSignalState
    via SignalStateRegistry. This adapter reads from that mirror, eliminating billable Temporal queries entirely.
    """

    def __init__(self, *, workflow_id: str) -> None:
        """
        Initialize Temporal signal adapter.
        """

        self.__workflow_id = workflow_id
        self.__state: WorkflowSignalState = SignalStateRegistry.shared().get(
            workflow_id=workflow_id,
        )

        logger.info(f"[signal-adapter] workflow={workflow_id} event=initialized mode=in_process")

    @property
    def workflow_id(self) -> str:
        """
        Get the workflow ID.
        """

        return self.__workflow_id

    def supports_interruption(self) -> bool:
        """
        Return interruption support for this adapter.
        """

        return True

    async def check_signal(self) -> Optional[str]:
        """
        Check for active control signal.

        Returns:
            SignalType value if present, else None.
        """

        self.__state.metrics.signal_checks += 1

        if self.__state.cancelled:
            logger.info(
                f"[signal-adapter] workflow={self.__workflow_id} event=check_signal result=CANCELLED"
            )
            return SignalType.CANCELLED.value

        if self.__state.paused:
            logger.info(
                f"[signal-adapter] workflow={self.__workflow_id} event=check_signal result=ASK"
            )
            return SignalType.ASK.value

        return None

    async def is_pause_requested(self) -> bool:
        """
        Check if pause is currently requested.
        """

        paused = self.__state.paused

        if paused:
            logger.info(
                f"[signal-adapter] workflow={self.__workflow_id} event=is_pause_requested result=true"
            )

        return paused

    async def wait_for_pause(self) -> None:
        """
        Block until a pause signal is received.
        """

        self.__state.metrics.pause_waits += 1
        logger.info(
            f"[signal-adapter] workflow={self.__workflow_id} event=wait_for_pause phase=entering"
        )

        while not self.__state.paused:
            await asyncio.to_thread(
                self.__state.wait_until,
                timeout=SIGNAL_HEARTBEAT_INTERVAL,
                predicate=lambda: self.__state.paused or self.__state.cancelled,
            )

            with contextlib.suppress(RuntimeError):
                self.__state.metrics.heartbeats_sent += 1
                activity.heartbeat("Running - waiting for pause signal")

        logger.info(
            f"[signal-adapter] workflow={self.__workflow_id} event=wait_for_pause phase=resolved"
        )

    async def wait_for_resume(self) -> None:
        """
        Block until a resume signal is received.
        """

        self.__state.metrics.resume_waits += 1
        logger.info(
            f"[signal-adapter] workflow={self.__workflow_id} event=wait_for_resume phase=entering"
        )

        while self.__state.paused:
            await asyncio.to_thread(
                self.__state.wait_until,
                timeout=SIGNAL_HEARTBEAT_INTERVAL,
                predicate=lambda: not self.__state.paused,
            )

            with contextlib.suppress(RuntimeError):
                self.__state.metrics.heartbeats_sent += 1
                activity.heartbeat("Paused - waiting for resume")

        logger.info(
            f"[signal-adapter] workflow={self.__workflow_id} event=wait_for_resume phase=resolved"
        )

    async def ask(self, *, prompt: str) -> str:
        """
        Request human input and block until context is injected.
        """

        logger.info(
            f'[signal-adapter] workflow={self.__workflow_id} event=ask phase=entering prompt="{prompt[:80]}"'
        )

        while True:
            if self.__state.has_context():
                context = self.__state.dequeue_context()
                if context is not None:
                    logger.info(
                        f"[signal-adapter] workflow={self.__workflow_id} event=ask phase=resolved "
                        f"context_length={len(context)}"
                    )
                    return context

            await asyncio.to_thread(
                self.__state.wait_until,
                timeout=SIGNAL_HEARTBEAT_INTERVAL,
                predicate=self.__state.has_context,
            )

            with contextlib.suppress(RuntimeError):
                self.__state.metrics.heartbeats_sent += 1
                activity.heartbeat(f"Waiting for human: {prompt[:50]}...")

    async def get_injected_context(self) -> Optional[str]:
        """
        DEPRECATED: Use peek_next_context and consume_context.
        Retrieve and consume injected user context.
        """

        context = self.__state.dequeue_context()

        if context is not None:
            logger.warning(
                f"[signal-adapter] workflow={self.__workflow_id} event=get_injected_context "
                f"deprecated=true context_length={len(context)}"
            )

        return context

    async def peek_next_context(self) -> Optional[str]:
        """
        Peek at the next context without consuming it.
        """

        return self.__state.peek_context()

    async def consume_context(self) -> None:
        """
        Explicitly consume the next context.
        """

        consumed = self.__state.dequeue_context()

        if consumed is not None:
            logger.debug(
                f"[signal-adapter] workflow={self.__workflow_id} event=consume_context "
                f"context_length={len(consumed)}"
            )

    async def has_injected_context(self) -> bool:
        """
        Check if there is injected context available.
        """

        return self.__state.has_context()
