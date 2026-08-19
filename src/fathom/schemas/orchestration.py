from __future__ import annotations

import time
import uuid
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from fathom.schemas.results import WorkflowResult
from fathom.schemas.run import RealignmentPolicy
from fathom.schemas.steps import StepResult

__all__ = [
    "ExecutionContext",
    "ExecutionRoadmap",
    "RealignmentPolicy",
    "RunnerConfig",
    "RunnerResult",
    "StepContext",
]


class StepContext(BaseModel):
    """Context for a single step execution."""

    step_id: str = Field(..., description="Unique identifier for the step")
    step_number: int = Field(..., description="Sequential number of the step")
    start_time: float = Field(..., description="Start timestamp of the step")
    end_time: Optional[float] = Field(default=None, description="End timestamp of the step")

    success: bool = Field(default=False, description="Whether the step executed successfully")
    error: Optional[str] = Field(default=None, description="Error message if step failed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional step metadata")

    @property
    def duration(self) -> float:
        """Step duration in seconds."""

        if self.end_time is None:
            return time.time() - self.start_time

        return self.end_time - self.start_time

    def complete(self, success: bool, error: Optional[str] = None) -> None:
        """Mark step as complete."""

        self.end_time = time.time()

        self.error = error
        self.success = success


class ExecutionContext(BaseModel):
    """
    Context for tracking workflow execution.

    Carries run-level metadata, timed step tracking, correlation IDs for tracing,
    and serializable state for checkpointing.
    """

    workflow_id: str = Field(..., description="ID of the workflow being executed")
    run_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())[:8],
        description="Unique ID for this specific run",
    )

    end_time: Optional[float] = Field(default=None, description="End timestamp of the execution")
    start_time: float = Field(
        default_factory=time.time, description="Start timestamp of the execution"
    )

    parent_id: Optional[str] = Field(default=None, description="Parent execution ID if nested")
    correlation_id: Optional[str] = Field(default=None, description="Correlation ID for tracing")

    metadata: Dict[str, Any] = Field(default_factory=dict, description="Execution-level metadata")
    steps: List[StepContext] = Field(default_factory=list, description="List of executed steps")

    @property
    def elapsed(self) -> float:
        """Elapsed time in seconds."""

        end = self.end_time or time.time()
        return end - self.start_time

    @property
    def steps_executed(self) -> int:
        """Number of steps executed."""

        return len(self.steps)

    @property
    def successful_steps(self) -> int:
        """Number of successful steps."""

        return sum(1 for step in self.steps if step.success)

    @property
    def failed_steps(self) -> int:
        """Number of failed steps."""

        return sum(1 for step in self.steps if not step.success and step.end_time is not None)

    def start_step(self, step_number: int) -> StepContext:
        """
        Start a new step.

        Args:
            step_number: Sequential step number.

        Returns:
            StepContext for the new step.
        """

        step = StepContext(
            start_time=time.time(),
            step_number=step_number,
            step_id=f"{self.run_id}-{step_number:03d}",
        )
        self.steps.append(step)

        return step

    def complete_step(
        self,
        step: StepContext,
        result: StepResult,
    ) -> None:
        """
        Complete a step with its result.

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
        """
        Get step execution history.

        Returns:
            List of step dictionaries.
        """

        return [
            {
                "error": step.error,
                "success": step.success,
                "step_id": step.step_id,
                "metadata": step.metadata,
                "step_number": step.step_number,
                "duration_seconds": step.duration,
            }
            for step in self.steps
        ]

    def get_history_summary(self) -> List[str]:
        """Get summary of previous actions."""

        return [
            f"Step {step.step_number}: {step.metadata.get('action', 'unknown')} "
            f"({'Success' if step.success else 'Failed'})"
            for step in self.steps
        ]

    def to_checkpoint(self) -> Dict[str, Any]:
        """
        Serialize to checkpoint.

        Returns:
            Serializable checkpoint dictionary.
        """

        return {
            "run_id": self.run_id,
            "metadata": self.metadata,
            "parent_id": self.parent_id,
            "start_time": self.start_time,
            "workflow_id": self.workflow_id,
            "elapsed_seconds": self.elapsed,
            "failed_steps": self.failed_steps,
            "correlation_id": self.correlation_id,
            "steps_executed": self.steps_executed,
            "successful_steps": self.successful_steps,
        }

    @classmethod
    def from_checkpoint(cls, data: Dict[str, Any]) -> "ExecutionContext":
        """
        Restore from checkpoint.

        Args:
            data: Checkpoint data.

        Returns:
            Restored ExecutionContext.
        """

        return cls(
            parent_id=data.get("parent_id"),
            run_id=str(data.get("run_id", "")),
            workflow_id=str(data["workflow_id"]),
            metadata=dict(data.get("metadata", {})),
            correlation_id=data.get("correlation_id"),
            start_time=float(data.get("start_time", time.time())),
        )


class ExecutionRoadmap(BaseModel):
    """
    An intent paired with the ordered high-level steps planned to achieve it.
    """

    intent: str = Field(..., description="Goal intent")
    steps: List[str] = Field(default_factory=list, description="Planned steps")


class RunnerConfig(BaseModel):
    """
    Workflow runner settings: whether to checkpoint state and the callback that persists it.
    """

    model_config = ConfigDict(frozen=True)

    enable_checkpoints: bool = Field(default=True, description="Enable state persistence")
    checkpoint_callback: Optional[Callable[[Dict[str, Any]], None]] = Field(
        default=None, description="Callback for saving state"
    )
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = Field(
        default=None, description="Callback for reporting progress"
    )
    enable_logging: bool = Field(default=True, description="Enable execution logging")


class RunnerResult(BaseModel):
    """Result from workflow runner."""

    model_config = ConfigDict(frozen=True)

    workflow_result: WorkflowResult = Field(..., description="Outcome of the workflow")
    execution_context: ExecutionContext = Field(..., description="Details of the execution run")

    checkpoints_saved: int = Field(default=0, description="Number of saved checkpoints")
    total_duration: float = Field(default=0.0, description="Total run time in seconds")
