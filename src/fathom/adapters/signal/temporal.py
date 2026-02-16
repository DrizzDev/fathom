"""Temporal-based signal adapter for HITL in distributed workflows."""

from __future__ import annotations

import asyncio
from typing import Optional
from logging import getLogger

from temporalio import activity

from fathom.interfaces.signal import SignalPort
from fathom.constants import SignalType

logger = getLogger(__name__)


class TemporalSignalAdapter(SignalPort):
    """
    Signal adapter that receives signals from Temporal workflows.
    
    This adapter queries the Temporal workflow state to detect signals.
    When the user sends a signal (pause/resume/inject/cancel) via HTTP API,
    the workflow updates its state, and this adapter detects it by querying.
    
    Architecture:
        User → HTTP API → Temporal Signal → Workflow State Update
                                                ↓
        Activity → Query Workflow State → Detect Change → Take Action
    
    Signal Flow:
        1. User sends POST /runs/{workflow_id}/pause
        2. Genymotion API calls client.signal_workflow("pause")
        3. Workflow's pause() signal handler sets self._paused = True
        4. Activity queries workflow.get_state() and sees paused=True
        5. Activity pauses execution and waits for resume
    
    Example:
        # In Temporal activity
        signal_adapter = TemporalSignalAdapter(workflow_id="my-workflow")
        
        # Build runner with this adapter
        builder = FathomBuilder(config)
        builder.signal(signal_adapter)
        runner = builder.build()
        
        # Execute - adapter polls workflow state for signals
        result = await runner.run_intent(intent="...")
    """

    def __init__(self, workflow_id: str) -> None:
        """
        Initialize Temporal signal adapter.
        
        Args:
            workflow_id: The Temporal workflow ID
        """
        self.__workflow_id = workflow_id
        self.__workflow_handle = None
        
        logger.info(f"TemporalSignalAdapter initialized for workflow {workflow_id}")

    async def __get_workflow_handle(self):
        """Get or create workflow handle for querying state."""
        if self.__workflow_handle is None:
            from temporalio import workflow
            self.__workflow_handle = workflow.get_external_workflow_handle(self.__workflow_id)
        return self.__workflow_handle

    async def __query_workflow_state(self) -> dict[str, bool]:
        """Query current workflow state."""
        try:
            handle = await self.__get_workflow_handle()
            from fathom.runtime.temporal.workflow import FathomWorkflow
            state = await handle.query(FathomWorkflow.get_state)
            return state
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
            return SignalType.CANCEL.value
        
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
            state = asyncio.run(self.__query_workflow_state())
            return state.get("paused", False)
        except Exception:
            return False

    async def wait_for_resume(self) -> None:
        """Block until RESUME signal received from Temporal."""
        logger.info(f"Workflow {self.__workflow_id} paused, waiting for resume signal")
        
        while True:
            state = await self.__query_workflow_state()
            
            if not state.get("paused"):
                logger.info(f"Workflow {self.__workflow_id} resumed")
                break
            
            try:
                activity.heartbeat("Paused - waiting for resume")
            except RuntimeError:
                pass
            
            await asyncio.sleep(0.5)

    async def request_input(self, *, prompt: str) -> str:
        """
        Request human input with prompt.
        
        In Temporal mode, this is not used because the user provides
        input via the /inject endpoint, not interactively.
        
        Returns:
            Empty string (not supported in Temporal mode)
        """
        logger.warning(
            f"request_input called in Temporal mode for workflow {self.__workflow_id} "
            "- not supported, use /inject endpoint instead"
        )
        return ""

    def get_injected_context(self) -> Optional[str]:
        """
        Get injected context from workflow.
        
        Returns:
            The injected context, or None if no context was injected
        """
        try:
            state = asyncio.run(self.__query_workflow_state())
            
            if state.get("has_context"):
                handle = asyncio.run(self.__get_workflow_handle())
                from fathom.runtime.temporal.workflow import FathomWorkflow
                context = asyncio.run(handle.query(FathomWorkflow.get_injected_context))
                return context
        except Exception as exception:
            logger.error(f"Failed to get injected context: {exception}")
        
        return None

    def has_injected_context(self) -> bool:
        """
        Check if there's injected context available.
        
        Returns:
            True if context was injected and not yet consumed
        """
        try:
            state = asyncio.run(self.__query_workflow_state())
            return state.get("has_context", False)
        except Exception:
            return False
    
    @property
    def workflow_id(self) -> str:
        """Get the workflow ID."""
        return self.__workflow_id
