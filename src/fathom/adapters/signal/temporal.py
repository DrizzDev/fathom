from __future__ import annotations

import asyncio
import contextlib
from logging import getLogger
from typing import Any, Dict, Optional

from temporalio import activity, workflow

from fathom.constants import SignalType
from fathom.interfaces.signal import SignalPort
from fathom.runtime.temporal.workflow import FathomWorkflow

logger = getLogger(__name__)


class TemporalSignalAdapter(SignalPort):
    """
    Signal adapter that receives signals from Temporal workflows.
    """

    def __init__(self, workflow_id: str) -> None:
        """
        Initialize Temporal signal adapter.

        Args:
            workflow_id: The Temporal workflow ID
        """

        self.__workflow_id = workflow_id
        self.__workflow_handle: Optional[Any] = None

        logger.info(f"TemporalSignalAdapter initialized for workflow {workflow_id}")

    @property
    def workflow_id(self) -> str:
        """
        Get the workflow ID.
        """

        return self.__workflow_id

    async def __get_workflow_handle(self) -> Any:
        """
        Get or create workflow handle for querying state.
        """

        if self.__workflow_handle is None:
            self.__workflow_handle = workflow.get_external_workflow_handle(self.__workflow_id)

        return self.__workflow_handle

    async def __query_workflow_state(self) -> Dict[str, bool]:
        """
        Query current workflow state.
        """

        try:
            handle = await self.__get_workflow_handle()
            response = await handle.query(FathomWorkflow.get_state)

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

    def is_pause_requested(self) -> bool:
        """
        Check if pause is requested.

        Returns:
            True if pause was requested
        """

        try:
            # Note: This is synchronous call to async method, risky but kept for now
            # Better to make this method async in Port if possible.
            state = asyncio.run(self.__query_workflow_state())
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

    def get_injected_context(self) -> Optional[str]:
        """
        Get injected context from workflow.
        """

        try:
            state = asyncio.run(self.__query_workflow_state())

            if state.get("has_context"):
                handle = asyncio.run(self.__get_workflow_handle())
                context = asyncio.run(handle.query(FathomWorkflow.get_injected_context))

                return str(context)
        except Exception as exception:
            logger.error(f"Failed to get injected context: {exception}")

        return None

    def has_injected_context(self) -> bool:
        """
        Check if there's injected context available.
        """

        try:
            state = asyncio.run(self.__query_workflow_state())
            return state.get("has_context", False)
        except Exception:
            return False
