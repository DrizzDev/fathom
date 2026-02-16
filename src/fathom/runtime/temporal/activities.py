"""Temporal activities for Fathom execution."""

from __future__ import annotations

from logging import getLogger
from typing import Any, Dict

from temporalio import activity

logger = getLogger(__name__)


@activity.defn
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
    from fathom.adapters.signal.temporal import TemporalSignalAdapter
    from fathom.runtime.builder import FathomBuilder
    from fathom.runtime.config import FathomConfig
    
    activity.logger.info(
        f"Starting Fathom intent execution for workflow {workflow_id}"
    )
    
    # Extract configuration
    planner_config = request.get("planner_configuration", {})
    
    # Create Fathom config
    config = FathomConfig(
        device_id=request["session_id"],
        credentials_path=None,  # Will use credentials_json
        credentials_json=planner_config.get("credentials_json"),
        project_id=planner_config.get("project_id"),
        location=planner_config.get("location"),
        model=planner_config.get("model"),
        max_steps=request.get("max_steps", 100),
        use_xml=request.get("use_xml", False),
        interactive=True,  # Enable HITL
    )
    
    # Create Temporal signal adapter
    signal_adapter = TemporalSignalAdapter(workflow_id=workflow_id)
    
    # Build runner with Temporal signal adapter
    builder = FathomBuilder(config=config)
    builder.signal(signal_adapter)  # Use .signal() - accepts any SignalPort
    runner = builder.build()
    
    try:
        # Send heartbeat to indicate activity is alive
        activity.heartbeat("Starting execution")
        
        # Execute intent
        result = await runner.run_intent(
            intent=request["intent"],
            workflow_id=workflow_id,
        )
        
        # Send final heartbeat
        activity.heartbeat(f"Completed: {result.steps} steps")
        
        # Convert result to dict
        return {
            "success": result.success,
            "steps": result.steps,
            "duration": result.duration,
            "error": result.error,
            "metrics": result.metrics.to_dict() if result.metrics else None,
        }
        
    except Exception as e:
        activity.logger.exception(f"Fathom execution failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "steps": 0,
            "duration": 0,
            "metrics": None,
        }
        
    finally:
        await runner.cleanup()


@activity.defn
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
    from fathom.adapters.signal.temporal import TemporalSignalAdapter
    from fathom.runtime.builder import FathomBuilder
    from fathom.runtime.config import FathomConfig
    
    activity.logger.info(
        f"Starting Fathom exploration for workflow {workflow_id}"
    )
    
    # Extract configuration
    planner_config = request.get("planner_configuration", {})
    
    # Create Fathom config
    config = FathomConfig(
        device_id=request["session_id"],
        credentials_path=None,
        credentials_json=planner_config.get("credentials_json"),
        project_id=planner_config.get("project_id"),
        location=planner_config.get("location"),
        model=planner_config.get("model"),
        max_steps=request.get("max_steps", 100),
        use_xml=request.get("use_xml", False),
        interactive=True,
    )
    
    # Create Temporal signal adapter
    signal_adapter = TemporalSignalAdapter(workflow_id=workflow_id)
    
    # Build runner
    builder = FathomBuilder(config=config)
    builder.signal(signal_adapter)  # Use .signal() - accepts any SignalPort
    runner = builder.build()
    
    try:
        activity.heartbeat("Starting exploration")
        
        # Execute exploration
        result = await runner.run_exploration(workflow_id=workflow_id)
        
        activity.heartbeat(f"Completed: {result.steps} steps")
        
        return {
            "success": result.success,
            "steps": result.steps,
            "duration": result.duration,
            "error": result.error,
            "metrics": result.metrics.to_dict() if result.metrics else None,
        }
        
    except Exception as e:
        activity.logger.exception(f"Exploration failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "steps": 0,
            "duration": 0,
            "metrics": None,
        }
        
    finally:
        await runner.cleanup()
