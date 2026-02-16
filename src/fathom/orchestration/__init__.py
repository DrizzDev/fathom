from __future__ import annotations

from fathom.core.context.manager import ContextManager

# New architecture re-exports for backward compatibility
from fathom.core.execution.engine import ExecutionEngine

# Legacy exports (existing orchestration code)
from fathom.orchestration.executor import StepExecutor
from fathom.orchestration.runner.fathom import FathomRunner
from fathom.orchestration.runner.workflow import WorkflowRunner
from fathom.runtime.runner import FathomRunner as NewFathomRunner
from fathom.schemas.orchestration import ExecutionContext

__all__ = [
    # Legacy exports
    "ExecutionContext",
    "StepExecutor",
    "FathomRunner",
    "WorkflowRunner",
    # New architecture exports
    "ExecutionEngine",
    "ContextManager",
    "NewFathomRunner",
]
