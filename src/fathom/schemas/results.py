from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from fathom.constants import StrategyStatus, WorkflowStatus
from fathom.schemas.actions import Action
from fathom.schemas.screens import ScreenCapture
from fathom.schemas.steps import StepResult


class ActionResult(BaseModel):
    """
    Result from device action execution.
    """

    model_config = {"frozen": True}

    success: bool = Field(description="Whether action succeeded")
    duration: int = Field(ge=0, description="Execution duration in milliseconds")
    error: Optional[str] = Field(default=None, description="Error if failed")
    output: Optional[str] = Field(default=None, description="Command output if any")


class AnalysisResult(BaseModel):
    """
    Result from screen analysis.
    """

    model_config = {"frozen": True}

    action: Action = Field(description="Recommended action")
    alternatives: List[Action] = Field(default_factory=list, description="Alternative actions")
    reasoning: str = Field(description="Reasoning process")
    screen_description: str = Field(description="Description of screen content")
    is_goal_complete: bool = Field(default=False, description="Whether goal appears complete")


class StrategyResult(BaseModel):
    """
    Result from strategy execution.
    """

    model_config = {"frozen": True}

    status: StrategyStatus = Field(description="Execution status")
    step_result: Optional[StepResult] = Field(default=None, description="Step result if executed")
    message: str = Field(description="Status message")
    should_checkpoint: bool = Field(default=False, description="Whether to checkpoint")

    @property
    def is_terminal(self) -> bool:
        """
        Whether this result ends execution.
        """
        return self.status != StrategyStatus.CONTINUE


class WorkflowResult(BaseModel):
    """
    Result of workflow execution.
    """

    model_config = {"frozen": True}

    workflow_id: str = Field(description="Workflow identifier")
    status: WorkflowStatus = Field(description="Final status")
    steps_executed: int = Field(ge=0, description="Total steps executed")
    duration: float = Field(ge=0.0, description="Total duration in seconds")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata")
    step_results: List[StepResult] = Field(default_factory=list, description="Step history")

    @property
    def success(self) -> bool:
        """
        Whether workflow completed successfully.
        """
        return self.status == WorkflowStatus.COMPLETED

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize to dictionary.
        """
        return {
            "workflow_id": self.workflow_id,
            "status": self.status.value,
            "steps_executed": self.steps_executed,
            "duration_seconds": self.duration,
            "success": self.success,
            "error": self.error,
            "metadata": self.metadata,
        }


class IntentResult(BaseModel):
    """
    Result of intent workflow execution.
    """

    model_config = {"frozen": True}

    intent: str = Field(description="Goal intent")
    success: bool = Field(description="Whether intent was achieved")
    steps_taken: int = Field(ge=0, description="Steps executed")
    completion_reason: str = Field(description="Reason for completion/failure")
    final_screen: Optional[ScreenCapture] = Field(default=None, description="Final screen capture")
    metrics: Dict[str, Dict[str, float]] = Field(
        default_factory=dict, description="Execution metrics (e.g., timings)"
    )

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize to dictionary.
        """
        return {
            "intent": self.intent,
            "success": self.success,
            "steps_taken": self.steps_taken,
            "completion_reason": self.completion_reason,
        }


class ExplorationResult(BaseModel):
    """
    Result of exploration workflow execution.
    """

    model_config = {"frozen": True}

    unique_screens: int = Field(ge=0, description="Number of unique screens found")
    total_transitions: int = Field(ge=0, description="Number of transitions executed")
    total_actions: int = Field(ge=0, description="Total actions executed")
    coverage_percentage: float = Field(ge=0.0, le=100.0, description="Estimated coverage")
    discovered_activities: List[str] = Field(
        default_factory=list, description="Discovered activities"
    )
    screen_graph: Dict[str, Any] = Field(default_factory=dict, description="Graph representation")

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize to dictionary.
        """
        return {
            "unique_screens": self.unique_screens,
            "total_transitions": self.total_transitions,
            "total_actions": self.total_actions,
            "coverage_percentage": self.coverage_percentage,
            "discovered_activities": self.discovered_activities,
            "screen_graph": self.screen_graph,
        }
