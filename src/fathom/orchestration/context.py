"""Execution context for workflow and step tracking."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from fathom.schemas.steps import StepResult


@dataclass
class StepContext:
    """Context for a single step execution."""

    step_id: str
    step_number: int
    start_time: float
    end_time: Optional[float] = None
    success: bool = False
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        """Step duration in seconds."""
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time

    def complete(self, success: bool, error: Optional[str] = None) -> None:
        """Mark step as complete."""
        self.end_time = time.time()
        self.success = success
        self.error = error


@dataclass
class ExecutionContext:
    """Context for tracking workflow execution.

    Provides:
    - Run-level metadata (workflow_id, run_id, timestamps)
    - Step tracking with timing
    - Serializable state for checkpointing
    - Correlation IDs for distributed tracing

    Thread-safe for async operations.
    """

    workflow_id: str
    run_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    parent_id: Optional[str] = None
    correlation_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    __steps: List[StepContext] = field(default_factory=list)

    @property
    def elapsed(self) -> float:
        """Elapsed time in seconds."""
        end = self.end_time or time.time()
        return end - self.start_time

    @property
    def steps_executed(self) -> int:
        """Number of steps executed."""
        return len(self.__steps)

    @property
    def successful_steps(self) -> int:
        """Number of successful steps."""
        return sum(1 for s in self.__steps if s.success)

    @property
    def failed_steps(self) -> int:
        """Number of failed steps."""
        return sum(1 for s in self.__steps if not s.success and s.end_time is not None)

    def start_step(self, step_number: int) -> StepContext:
        """Start a new step.

        Args:
            step_number: Sequential step number.

        Returns:
            StepContext for the new step.
        """
        step = StepContext(
            step_id=f"{self.run_id}-{step_number:03d}",
            step_number=step_number,
            start_time=time.time(),
        )
        self.__steps.append(step)
        return step

    def complete_step(
        self,
        step: StepContext,
        result: StepResult,
    ) -> None:
        """Complete a step with its result.

        Args:
            step: Step context to complete.
            result: Step execution result.
        """
        step.complete(result.success, result.error)
        step.metadata["screen_changed"] = result.screen_changed
        step.metadata["action"] = result.step.action.to_description()

    def finish(self) -> None:
        """Mark execution as finished."""
        self.end_time = time.time()

    def get_step_history(self) -> List[Dict[str, Any]]:
        """Get step execution history.

        Returns:
            List of step dictionaries.
        """
        return [
            {
                "step_id": s.step_id,
                "step_number": s.step_number,
                "duration_seconds": s.duration,
                "success": s.success,
                "error": s.error,
                "metadata": s.metadata,
            }
            for s in self.__steps
        ]

    def to_checkpoint(self) -> Dict[str, Any]:
        """Serialize to checkpoint.

        Returns:
            Serializable checkpoint dictionary.
        """
        return {
            "workflow_id": self.workflow_id,
            "run_id": self.run_id,
            "start_time": self.start_time,
            "elapsed_seconds": self.elapsed,
            "steps_executed": self.steps_executed,
            "successful_steps": self.successful_steps,
            "failed_steps": self.failed_steps,
            "parent_id": self.parent_id,
            "correlation_id": self.correlation_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_checkpoint(cls, data: Dict[str, Any]) -> "ExecutionContext":
        """Restore from checkpoint.

        Args:
            data: Checkpoint data.

        Returns:
            Restored ExecutionContext.
        """
        return cls(
            workflow_id=str(data["workflow_id"]),
            run_id=str(data.get("run_id", "")),
            start_time=float(data.get("start_time", time.time())),
            parent_id=data.get("parent_id"),
            correlation_id=data.get("correlation_id"),
            metadata=dict(data.get("metadata", {})),
        )
