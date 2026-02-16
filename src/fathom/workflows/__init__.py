from __future__ import annotations

# Legacy exports (existing workflow code)
from fathom.schemas.configuration import WorkflowConfig
from fathom.schemas.results import WorkflowResult

# New architecture re-exports for backward compatibility
from fathom.strategies.exploration import ExplorationStrategy
from fathom.strategies.intent import IntentStrategy
from fathom.workflows.base import BaseWorkflow
from fathom.workflows.exploration import ExplorationWorkflow
from fathom.workflows.intent import IntentWorkflow

__all__ = [
    # Legacy exports
    "BaseWorkflow",
    "ExplorationWorkflow",
    "IntentWorkflow",
    "WorkflowConfig",
    "WorkflowResult",
    # New architecture exports
    "ExplorationStrategy",
    "IntentStrategy",
]
