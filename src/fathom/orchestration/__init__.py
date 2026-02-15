from __future__ import annotations

# Legacy exports (existing orchestration code)
from fathom.orchestration.executor import StepExecutor
from fathom.orchestration.runner.fathom import FathomRunner
from fathom.orchestration.runner.workflow import WorkflowRunner
from fathom.schemas.orchestration import ExecutionContext

# New architecture re-exports for backward compatibility
from fathom.core.execution.engine import ExecutionEngine
from fathom.core.context.manager import ContextManager
from fathom.runtime.runner import FathomRunner as NewFathomRunner

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
