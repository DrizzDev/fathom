"""
Temporal workflow integration for Fathom.

This module provides Temporal workflow wrappers for Fathom execution,
enabling distributed, long-running, and HITL-capable mobile automation.

Installation:
    pip install fathom[temporal]

Usage:
    from fathom.runtime.temporal import FathomWorkflow

    # Register with Temporal worker
    worker = Worker(
        client,
        task_queue="fathom-tasks",
        workflows=[FathomWorkflow],
        activities=[execute_fathom_intent, execute_fathom_exploration],
    )
"""

from __future__ import annotations

# Graceful import handling for optional Temporal dependency
try:
    from temporalio import workflow

    TEMPORAL_AVAILABLE = True
except ImportError:
    TEMPORAL_AVAILABLE = False
    workflow = None

if TEMPORAL_AVAILABLE:
    from .activities import execute_fathom_exploration, execute_fathom_intent
    from .workflow import FathomWorkflow

    __all__ = [
        "FathomWorkflow",
        "execute_fathom_intent",
        "execute_fathom_exploration",
        "TEMPORAL_AVAILABLE",
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
