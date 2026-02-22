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

    def __init__(self, workflow_id: str, namespace: str = "default") -> None:
        """
        Initialize Temporal signal adapter.

        Args:
            workflow_id: The Temporal workflow ID
            namespace: The Temporal namespace
        """

        self.__workflow_id = workflow_id
        self.__namespace = namespace
        self.__client: Optional[Client] = None
        self.__workflow_handle: Optional[Any] = None

        logger.info(
            f"TemporalSignalAdapter initialized for workflow {workflow_id} (ns: {namespace})"
        )

    @property
    def workflow_id(self) -> str:
        """
        Get the workflow ID.
        """

        return self.__workflow_id

    async def __get_workflow_handle(self) -> Any:
        """
        Get or create workflow handle for querying state using a clean client.
        """

        if self.__client is None:
            # We connect a fresh client using the cluster address.
            # This bypasses activity sandbox restrictions on the current task's handle.
            import os

            target = os.getenv("TEMPORAL_HOST", "localhost:7233")

            self.__client = await Client.connect(
                target=target,
                namespace=self.__namespace,
            )

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
        Request human input with prompt.
        """

        _ = prompt

        logger.warning(
            f"ask called in Temporal mode for workflow {self.__workflow_id} "
            "- not supported, use /inject endpoint instead"
        )
        return ""

    async def get_injected_context(self) -> Optional[str]:
        """
        Get injected context from workflow via external handle.
        """

        try:
            state = await self.__query_workflow_state()

            if state.get("has_context"):
                handle = await self.__get_workflow_handle()
                # Use string query name to avoid workflow context triggers
                context = await handle.query("get_injected_context")

                return str(context)
        except Exception as exception:
            logger.error(f"Failed to get injected context: {exception}")

        return None

    async def has_injected_context(self) -> bool:
        """
        Check if there's injected context available.
        """

        try:
            state = await self.__query_workflow_state()
            return state.get("has_context", False)
        except Exception:
            return False
