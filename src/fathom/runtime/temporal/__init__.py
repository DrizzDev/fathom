"""
Temporal workflow integration for Fathom.

This module provides Temporal workflow wrappers for Fathom execution,
enabling distributed, long-running, and HITL-capable mobile automation.

Installation:
    pip install fathom[temporal]

Usage:
    from fathom.runtime.temporal import FathomWorkflow, FathomActivities

    # Register with Temporal worker
    activities = FathomActivities()
    worker = Worker(
        client,
        task_queue="fathom-tasks",
        workflows=[FathomWorkflow],
        activities=[activities.execute_intent, activities.execute_exploration],
    )
"""

from __future__ import annotations

# Graceful import handling for optional Temporal dependency
try:
    from temporalio import workflow

    TEMPORAL_AVAILABLE = bool(workflow)
except ImportError:
    workflow = None
    TEMPORAL_AVAILABLE = False

if TEMPORAL_AVAILABLE:
    from .activities import FathomActivities
    from .constants import WORKFLOW_PASSTHROUGH_MODULES
    from .state import SignalStateRegistry
    from .workflow import FathomWorkflow

    __all__ = [
        "FathomWorkflow",
        "FathomActivities",
        "TEMPORAL_AVAILABLE",
        "SignalStateRegistry",
        "WORKFLOW_PASSTHROUGH_MODULES",
    ]
else:
    __all__ = ["TEMPORAL_AVAILABLE"]


def check_temporal_available() -> None:
    """
    Check if Temporal is available and raise helpful error if not.

    Raises:
        ImportError: If temporalio is not installed
    """

    if not TEMPORAL_AVAILABLE:
        raise ImportError(
            "Temporal integration requires the 'temporalio' package. "
            "Install it with: pip install fathom[temporal]"
        )
