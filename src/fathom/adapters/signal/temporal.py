from __future__ import annotations

import asyncio
import contextlib
from logging import getLogger
from typing import Any, Dict, Optional

from temporalio import activity
from temporalio.client import Client

from fathom.constants import SignalType
from fathom.interfaces.signal import SignalPort

logger = getLogger(__name__)


class TemporalSignalAdapter(SignalPort):
    """
    Signal adapter that receives signals from Temporal workflows.
    Uses an external client to bypass activity-restricted workflow context.
    """

    def __init__(
        self,
        namespace: str,
        target_host: str,
        workflow_id: str,
        api_key: Optional[str] = None,
    ) -> None:
        """
        Initialize Temporal signal adapter.
        """

        self.__api_key = api_key
        self.__namespace = namespace
        self.__target_host = target_host

        self.__workflow_id = workflow_id
        self.__client: Optional[Client] = None

        self.__connection_lock = asyncio.Lock()
        self.__workflow_handle: Optional[Any] = None

        logger.info(
            f"TemporalSignalAdapter initialized for workflow {workflow_id} "
            f"(ns: {namespace}, host: {target_host})"
        )

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

    async def __get_workflow_handle(self) -> Any:
        """
        Get or create workflow handle for querying state using an isolated client.
        """

        async with self.__connection_lock:
            if self.__client is None:
                # We connect an isolated client using the cluster address.
                # This bypasses activity sandbox restrictions on the current task's handle.

                # Align with genymotion_project/services/temporal/client.py
                if "localhost" not in self.__target_host and "127.0.0.1" not in self.__target_host:
                    from temporalio.service import TLSConfig

                    tls_configuration = TLSConfig()
                else:
                    tls_configuration = False

                try:
                    # Attempt connection with target_host
                    self.__client = await Client.connect(
                        tls=tls_configuration,
                        api_key=self.__api_key,
                        namespace=self.__namespace,
                        target_host=self.__target_host,
                    )
                except (TypeError, ValueError):
                    # Fallback if SDK version is different or target_host not available
                    try:
                        self.__client = await Client.connect(
                            tls=tls_configuration,
                            api_key=self.__api_key,
                            namespace=self.__namespace,
                            target_host=self.__target_host,
                        )
                    except Exception as e:
                        logger.error(f"Failed to connect to Temporal at {self.__target_host}: {e}")
                        raise

            if self.__client is None:
                raise RuntimeError("Failed to establish Temporal client connection")

            if self.__workflow_handle is None:
                self.__workflow_handle = self.__client.get_workflow_handle(self.__workflow_id)

        return self.__workflow_handle

    async def __query_workflow_state(self) -> Dict[str, bool]:
        """
        Query current workflow state via external handle.
        """

        try:
            handle = await self.__get_workflow_handle()
            # Use string name for query to avoid importing FathomWorkflow (circular dep)
            # and to ensure it's treated as a pure external query.
            response = await handle.query("get_state")

            return dict(response)
        except Exception as exception:
            logger.error(f"Failed to query workflow state: {exception}")
            return {"paused": False, "cancelled": False, "has_context": False}

    async def check_signal(self) -> Optional[str]:
        """
        Check for control signal from Temporal workflow.

        Returns:
            SignalType.ASK if pause requested, SignalType.CANCEL if cancelled, None otherwise
        """

        state = await self.__query_workflow_state()

        if state.get("cancelled"):
            return SignalType.CANCELLED.value

        if state.get("paused"):
            return SignalType.ASK.value

        return None

    async def is_pause_requested(self) -> bool:
        """
        Check if pause is requested.

        Returns:
            True if pause was requested
        """

        try:
            state = await self.__query_workflow_state()
            return state.get("paused", False)
        except Exception:
            return False

    async def wait_for_pause(self) -> None:
        """
        Block until a pause signal is received from Temporal.
        """

        while True:
            state = await self.__query_workflow_state()

            if state.get("paused"):
                logger.info(f"Workflow {self.__workflow_id} pause signal detected")
                return

            with contextlib.suppress(RuntimeError):
                activity.heartbeat("Running - waiting for pause signal")

            # Check frequently enough to be responsive but not flood queries
            await asyncio.sleep(0.5)

    async def wait_for_resume(self) -> None:
        """
        Block until RESUME signal received from Temporal.
        """

        logger.info(f"Workflow {self.__workflow_id} paused, waiting for resume signal")

        while True:
            state = await self.__query_workflow_state()

            if not state.get("paused"):
                logger.info(f"Workflow {self.__workflow_id} resumed")
                break

            with contextlib.suppress(RuntimeError):
                activity.heartbeat("Paused - waiting for resume")

            await asyncio.sleep(0.5)

    async def ask(self, *, prompt: str) -> str:
        """
        Request human input with prompt and wait for injected context.
        """

        logger.info(f"Blocking for human assistance: {prompt}")

        # 1. Wait for context to be injected via /inject endpoint
        # We poll the workflow state until has_context is True
        while True:
            state = await self.__query_workflow_state()

            if state.get("has_context"):
                # Use peek + consume to ensure we don't lose data
                context = await self.peek_next_context()
                if context:
                    await self.consume_context()
                    logger.info(f"Human assistance received: {context}")
                    return context

            # Heartbeat to prevent activity timeout while waiting for human
            with contextlib.suppress(RuntimeError):
                activity.heartbeat(f"Waiting for human: {prompt[:50]}...")

            await asyncio.sleep(1.0)

    async def get_injected_context(self) -> Optional[str]:
        """
        DEPRECATED: Use peek_next_context and consume_context.
        Get injected context from workflow via external handle.
        """

        context = await self.peek_next_context()
        if context:
            await self.consume_context()

        return context

    async def peek_next_context(self) -> Optional[str]:
        """
        Peek at the next context without consuming it.
        """

        try:
            handle = await self.__get_workflow_handle()
            context = await handle.query("peek_next_context")
            return str(context) if context else None
        except Exception as exception:
            logger.error(f"Failed to peek context: {exception}")
            return None

    async def consume_context(self) -> None:
        """
        Explicitly consume the next context via signal.
        """

        try:
            handle = await self.__get_workflow_handle()
            await handle.signal("consume_context")
        except Exception as exception:
            logger.error(f"Failed to consume context: {exception}")

    async def has_injected_context(self) -> bool:
        """
        Check if there's injected context available.
        """

        try:
            state = await self.__query_workflow_state()
            return state.get("has_context", False)
        except Exception:
            return False
