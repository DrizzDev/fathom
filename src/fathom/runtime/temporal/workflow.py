from __future__ import annotations

from datetime import timedelta
from typing import Any, Dict, Optional

from temporalio import workflow
from temporalio.common import RetryPolicy


class FathomBaseWorkflow:
    """
    Base state and signals for Fathom workflows.
    """

    def __init__(self) -> None:
        self.__paused = False
        self.__cancelled = False
        self.__injected_context: Optional[str] = None

    @workflow.signal  # type: ignore[untyped-decorator]
    async def pause(self) -> None:
        """
        Signal to pause execution.
        """

        workflow.logger.info("Received pause signal")
        self.__paused = True

    @workflow.signal  # type: ignore[untyped-decorator]
    async def resume(self) -> None:
        """
        Signal to resume execution.
        """

        workflow.logger.info("Received resume signal")
        self.__paused = False

    @workflow.signal  # type: ignore[untyped-decorator]
    async def inject(self, context: str) -> None:
        """
        Signal to inject user context.
        """

        workflow.logger.info(f"Received inject signal with context: {context}")
        self.__injected_context = context

    @workflow.signal  # type: ignore[untyped-decorator]
    async def cancel(self) -> None:
        """
        Signal to cancel execution.
        """

        workflow.logger.info("Received cancel signal")
        self.__cancelled = True

    @workflow.query  # type: ignore[untyped-decorator]
    def get_state(self) -> Dict[str, Any]:
        """
        Query current workflow state.
        """

        return {
            "paused": self.__paused,
            "cancelled": self.__cancelled,
            "has_context": self.__injected_context is not None,
        }

    @workflow.query  # type: ignore[untyped-decorator]
    def get_injected_context(self) -> Optional[str]:
        """
        Query and clear injected context.
        """

        context = self.__injected_context
        self.__injected_context = None

        return context


@workflow.defn(name="FathomWorkflow")
class FathomWorkflow(FathomBaseWorkflow):
    """
    Temporal workflow for executing Fathom intent tasks.
    """

    @workflow.run  # type: ignore[untyped-decorator]
    async def run(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute Fathom intent with HITL support.
        """

        workflow.logger.info(
            f"Starting Fathom intent workflow for session {request.get('session_id')} "
            f"with intent: {request.get('intent')}"
        )

        try:
            result = await workflow.execute_activity(
                activity="EXECUTE_INTENT",
                args=[workflow.info().workflow_id, request],
                start_to_close_timeout=timedelta(minutes=30),
                heartbeat_timeout=timedelta(seconds=30),
                retry_policy=RetryPolicy(maximum_attempts=1),
            )

            workflow.logger.info(
                f"Workflow completed successfully: {result.get('steps')} steps in "
                f"{result.get('duration')}ms"
            )
            return dict(result)

        except Exception as exception:
            workflow.logger.exception(f"Workflow failed: {exception}")
            return {
                "steps": 0,
                "duration": 0,
                "metrics": None,
                "success": False,
                "error": str(exception),
            }


@workflow.defn(name="FathomExplorationWorkflow")
class FathomExplorationWorkflow(FathomBaseWorkflow):
    """
    Temporal workflow for executing Fathom autonomous exploration.
    """

    @workflow.run  # type: ignore[untyped-decorator]
    async def run(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute Fathom exploration with HITL support.
        """

        workflow.logger.info(f"Starting Fathom exploration with payload {request}")

        try:
            result = await workflow.execute_activity(
                activity="EXECUTE_EXPLORATION",
                args=[workflow.info().workflow_id, request],
                start_to_close_timeout=timedelta(minutes=30),
                heartbeat_timeout=timedelta(seconds=30),
                retry_policy=RetryPolicy(maximum_attempts=1),
            )

            workflow.logger.info(f"Exploration completed: {result.get('steps')} steps")
            return dict(result)

        except Exception as exception:
            workflow.logger.exception(f"Exploration failed: {exception}")
            return {
                "steps": 0,
                "duration": 0,
                "metrics": None,
                "success": False,
                "error": str(exception),
            }
