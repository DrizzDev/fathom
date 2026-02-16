"""Temporal workflow for Fathom execution with HITL support."""

from __future__ import annotations

from datetime import timedelta
from typing import Any, Dict, Optional

from temporalio import workflow
from temporalio.common import RetryPolicy

from .activities import execute_fathom_exploration, execute_fathom_intent


@workflow.defn
class FathomWorkflow:
    """
    Temporal workflow for executing Fathom tasks with HITL support.
    """

    def __init__(self) -> None:
        """Initialize workflow state."""
        self.__paused = False
        self.__injected_context: Optional[str] = None
        self.__cancelled = False

    @workflow.run  # type: ignore[untyped-decorator]
    async def run(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute Fathom intent with HITL support.
        """
        workflow.logger.info(
            f"Starting Fathom workflow for session {request.get('session_id')} "
            f"with intent: {request.get('intent')}"
        )

        try:
            result = await workflow.execute_activity(
                execute_fathom_intent,
                args=[request, workflow.info().workflow_id],
                start_to_close_timeout=timedelta(minutes=30),
                heartbeat_timeout=timedelta(seconds=30),
                retry_policy=RetryPolicy(
                    maximum_attempts=1,
                ),
            )

            workflow.logger.info(
                f"Workflow completed successfully: {result.get('steps')} steps in "
                f"{result.get('duration')}ms"
            )
            return dict(result)

        except Exception as exception:
            workflow.logger.exception(f"Workflow failed: {exception}")
            return {
                "success": False,
                "error": str(exception),
                "steps": 0,
                "duration": 0,
                "metrics": None,
            }

    @workflow.run  # type: ignore[untyped-decorator]
    async def run_exploration(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute Fathom exploration with HITL support.
        """
        workflow.logger.info(f"Starting Fathom exploration for session {request.get('session_id')}")

        try:
            result = await workflow.execute_activity(
                execute_fathom_exploration,
                args=[request, workflow.info().workflow_id],
                start_to_close_timeout=timedelta(minutes=30),
                heartbeat_timeout=timedelta(seconds=30),
                retry_policy=RetryPolicy(
                    maximum_attempts=1,
                ),
            )

            workflow.logger.info(f"Exploration completed: {result.get('steps')} steps")
            return dict(result)

        except Exception as exception:
            workflow.logger.exception(f"Exploration failed: {exception}")
            return {
                "success": False,
                "error": str(exception),
                "steps": 0,
                "duration": 0,
                "metrics": None,
            }

    @workflow.signal  # type: ignore[untyped-decorator]
    async def pause(self) -> None:
        """Signal to pause execution."""
        workflow.logger.info("Received pause signal")
        self.__paused = True

    @workflow.signal  # type: ignore[untyped-decorator]
    async def resume(self) -> None:
        """Signal to resume execution."""
        workflow.logger.info("Received resume signal")
        self.__paused = False

    @workflow.signal  # type: ignore[untyped-decorator]
    async def inject(self, context: str) -> None:
        """Signal to inject user context/guidance."""
        workflow.logger.info(f"Received inject signal with context: {context}")
        self.__injected_context = context

    @workflow.signal  # type: ignore[untyped-decorator]
    async def cancel(self) -> None:
        """Signal to cancel execution."""
        workflow.logger.info("Received cancel signal")
        self.__cancelled = True

    @workflow.query  # type: ignore[untyped-decorator]
    def get_state(self) -> Dict[str, Any]:
        """Query current workflow state."""
        return {
            "paused": self.__paused,
            "cancelled": self.__cancelled,
            "has_context": self.__injected_context is not None,
        }

    @workflow.query  # type: ignore[untyped-decorator]
    def get_injected_context(self) -> Optional[str]:
        """Query and clear injected context."""
        context = self.__injected_context
        self.__injected_context = None
        return context
