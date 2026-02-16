"""Temporal workflow for Fathom execution with HITL support."""

from __future__ import annotations

from datetime import timedelta
from typing import Any, Dict

from temporalio import workflow
from temporalio.common import RetryPolicy

from .activities import execute_fathom_intent, execute_fathom_exploration


@workflow.defn
class FathomWorkflow:
    """
    Temporal workflow for executing Fathom tasks with HITL support.
    
    This workflow wraps Fathom execution in a Temporal workflow, enabling:
    - Distributed execution across workers
    - Long-running tasks (hours/days)
    - HITL via signals (pause/resume/inject)
    - State persistence and recovery
    - Observability via Temporal UI
    
    Supported signals:
    - pause: Pause execution
    - resume: Resume execution  
    - inject: Inject user context/guidance
    - cancel: Cancel execution
    
    Example:
        # Start workflow
        await client.start_workflow(
            FathomWorkflow.run,
            args=[{
                "session_id": "emulator-5554",
                "intent": "Search for something",
                "planner_configuration": {...},
            }],
            id="unique-workflow-id",
            task_queue="fathom-tasks",
        )
        
        # Send signals
        await client.signal_workflow("unique-workflow-id", "pause")
        await client.signal_workflow("unique-workflow-id", "inject", "additional context")
        await client.signal_workflow("unique-workflow-id", "resume")
    """

    def __init__(self) -> None:
        """Initialize workflow state."""
        self._paused = False
        self._injected_context: str | None = None
        self._cancelled = False

    @workflow.run
    async def run(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute Fathom intent with HITL support.
        
        Args:
            request: Agent run request containing:
                - session_id: Device session ID (required)
                - intent: User intent (required)
                - enricher_url: URL for enrichment service (optional)
                - planner_configuration: LLM config with:
                    - model: Model name (e.g., "vertex_ai/gemini-2.0-flash-exp")
                    - credentials_json: GCP credentials JSON string
                    - project_id: GCP project ID
                    - location: GCP location (e.g., "us-central1")
                - max_steps: Maximum steps (optional, default: 100)
                - use_xml: Use XML hierarchy (optional, default: False)
        
        Returns:
            Execution result with:
                - success: Whether execution succeeded
                - steps: Number of steps executed
                - duration: Duration in milliseconds
                - error: Error message if failed
                - metrics: Execution metrics (tokens, timing, etc.)
        """
        workflow.logger.info(
            f"Starting Fathom workflow for session {request.get('session_id')} "
            f"with intent: {request.get('intent')}"
        )
        
        try:
            # Execute Fathom as a Temporal activity
            # Activity runs in a worker and can be monitored/cancelled
            result = await workflow.execute_activity(
                execute_fathom_intent,
                args=[request, workflow.info().workflow_id],
                start_to_close_timeout=timedelta(minutes=30),
                heartbeat_timeout=timedelta(seconds=30),
                retry_policy=RetryPolicy(
                    maximum_attempts=1,  # No retries for HITL workflows
                ),
            )
            
            workflow.logger.info(
                f"Workflow completed successfully: {result.get('steps')} steps in "
                f"{result.get('duration')}ms"
            )
            return result
            
        except Exception as e:
            workflow.logger.exception(f"Workflow failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "steps": 0,
                "duration": 0,
                "metrics": None,
            }

    @workflow.run
    async def run_exploration(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute Fathom exploration with HITL support.
        
        Args:
            request: Exploration request containing:
                - session_id: Device session ID (required)
                - enricher_url: URL for enrichment service (optional)
                - planner_configuration: LLM config (same as run())
                - max_steps: Maximum steps (optional, default: 100)
                - use_xml: Use XML hierarchy (optional, default: False)
        
        Returns:
            Execution result (same format as run())
        """
        workflow.logger.info(
            f"Starting Fathom exploration for session {request.get('session_id')}"
        )
        
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
            return result
            
        except Exception as e:
            workflow.logger.exception(f"Exploration failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "steps": 0,
                "duration": 0,
                "metrics": None,
            }

    @workflow.signal
    async def pause(self) -> None:
        """
        Signal to pause execution.
        
        The activity will detect this and pause at the next safe point
        (typically before the next LLM call).
        """
        workflow.logger.info("Received pause signal")
        self._paused = True

    @workflow.signal
    async def resume(self) -> None:
        """
        Signal to resume execution.
        
        The activity will continue from where it paused.
        """
        workflow.logger.info("Received resume signal")
        self._paused = False

    @workflow.signal
    async def inject(self, context: str) -> None:
        """
        Signal to inject user context/guidance.
        
        Args:
            context: User-provided context, can be:
                - Guidance: "Wait for page to load"
                - Clarification: "The button is at bottom right"
                - Sub-goal: "First scroll down, then click"
                - Modified intent: "Actually search for X instead"
        """
        workflow.logger.info(f"Received inject signal with context: {context}")
        self._injected_context = context

    @workflow.signal
    async def cancel(self) -> None:
        """
        Signal to cancel execution.
        
        The activity will stop at the next safe point.
        """
        workflow.logger.info("Received cancel signal")
        self._cancelled = True

    @workflow.query
    def get_state(self) -> Dict[str, Any]:
        """
        Query current workflow state.
        
        Returns:
            Current state including paused status and injected context
        """
        return {
            "paused": self._paused,
            "cancelled": self._cancelled,
            "has_context": self._injected_context is not None,
        }
    @workflow.query
    def get_injected_context(self) -> str | None:
        """
        Query and clear injected context.

        Returns:
            The injected context, or None if no context
        """
        context = self._injected_context
        self._injected_context = None
        return context

