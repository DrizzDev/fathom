"""Temporal activities for Fathom execution."""

from __future__ import annotations

from logging import getLogger
from typing import Any, Dict

from temporalio import activity

logger = getLogger(__name__)


@activity.defn  # type: ignore[untyped-decorator]
async def execute_fathom_intent(
    request: Dict[str, Any],
    workflow_id: str,
) -> Dict[str, Any]:
    """
    Activity to execute Fathom intent.

    This runs as a Temporal activity so it can be monitored, cancelled,
    and send heartbeats to indicate progress.

    Args:
        request: Agent run request
        workflow_id: Temporal workflow ID for signal coordination

    Returns:
        Execution result dictionary
    """
    from fathom.adapters.device.adb import ADBDevice
    from fathom.adapters.llm.gemini import GeminiLLM
    from fathom.adapters.signal.temporal import TemporalSignalAdapter
    from fathom.runtime.builder import Fathom
    from fathom.schemas.configuration import GeminiConfig

    activity.logger.info(f"Starting Fathom intent execution for workflow {workflow_id}")

    # Extract configuration
    planner_config = request.get("planner_configuration", {})

    gemini_config = GeminiConfig(
        model=planner_config.get("model", "gemini-2.0-flash-exp"),
        project_id=planner_config.get("project_id"),
        location=planner_config.get("location"),
        credentials_path=None,  # Assume auth handled via environment or other means for now
    )

    # Create Temporal signal adapter
    signal_adapter = TemporalSignalAdapter(workflow_id=workflow_id)

    # Build runner using the fluent builder
    runner = (
        Fathom.builder()
        .device(device=ADBDevice(serial=request["session_id"]))
        .llm(llm=GeminiLLM(configuration=gemini_config))
        .signal(signal=signal_adapter)
        .build()
    )

    try:
        # Send heartbeat to indicate activity is alive
        activity.heartbeat("Starting execution")

        # Execute intent
        result = await runner.run_intent(
            intent=request["intent"],
            max_steps=request.get("max_steps", 20),
            use_xml=request.get("use_xml", False),
            request_id=workflow_id,
        )

        # Send final heartbeat
        activity.heartbeat(f"Completed: {result.steps_taken} steps")

        # Convert result to dict
        return {
            "success": result.success,
            "steps": result.steps_taken,
            "duration": result.duration,
            "error": result.error,
            "metrics": result.metrics if result.metrics else None,
        }

    except Exception as exception:
        activity.logger.exception(f"Fathom execution failed: {exception}")
        return {
            "success": False,
            "error": str(exception),
            "steps": 0,
            "duration": 0,
            "metrics": None,
        }

    finally:
        await runner.cleanup()


@activity.defn  # type: ignore[untyped-decorator]
async def execute_fathom_exploration(
    request: Dict[str, Any],
    workflow_id: str,
) -> Dict[str, Any]:
    """
    Activity to execute Fathom exploration.

    Args:
        request: Exploration request
        workflow_id: Temporal workflow ID

    Returns:
        Execution result dictionary
    """
    from fathom.adapters.device.adb import ADBDevice
    from fathom.adapters.llm.gemini import GeminiLLM
    from fathom.adapters.signal.temporal import TemporalSignalAdapter
    from fathom.runtime.builder import Fathom
    from fathom.schemas.configuration import GeminiConfig

    activity.logger.info(f"Starting Fathom exploration for workflow {workflow_id}")

    # Extract configuration
    planner_config = request.get("planner_configuration", {})

    gemini_config = GeminiConfig(
        model=planner_config.get("model", "gemini-2.0-flash-exp"),
        project_id=planner_config.get("project_id"),
        location=planner_config.get("location"),
    )

    # Create Temporal signal adapter
    signal_adapter = TemporalSignalAdapter(workflow_id=workflow_id)

    # Build runner
    runner = (
        Fathom.builder()
        .device(device=ADBDevice(serial=request["session_id"]))
        .llm(llm=GeminiLLM(configuration=gemini_config))
        .signal(signal=signal_adapter)
        .build()
    )

    try:
        activity.heartbeat("Starting exploration")

        # Execute exploration
        result = await runner.run_exploration(
            max_steps=request.get("max_steps", 50), request_id=workflow_id
        )

        activity.heartbeat(f"Completed: {result.steps_executed} steps")

        return {
            "success": result.success,
            "steps": result.steps_executed,
            "duration": result.duration,
            "error": result.error,
            "metrics": None,  # ExplorationResult doesn't have metrics yet in this schema
        }

    except Exception as exception:
        activity.logger.exception(f"Exploration failed: {exception}")
        return {
            "success": False,
            "error": str(exception),
            "steps": 0,
            "duration": 0,
            "metrics": None,
        }

    finally:
        await runner.cleanup()
