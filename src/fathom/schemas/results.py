from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from fathom.constants import StrategyStatus
from fathom.schemas.actions import Action
from fathom.schemas.steps import Step, StepResult


class AnalysisResult(BaseModel):
    """
    Result of vision analysis.
    """

    action: Action = Field(description="Primary recommended action")
    alternatives: List[Action] = Field(
        default_factory=list, description="Alternative actions considered"
    )
    reasoning: str = Field(description="Reasoning process")
    screen_description: str = Field(description="Description of screen content")
    is_goal_complete: bool = Field(
        default=False, description="Whether the user intent has been fully achieved"
    )
    memories: int = Field(
        default=0, description="Number of historical experiences retrieved for this state"
    )
    metrics: Dict[str, float] = Field(
        default_factory=dict, description="Internal analysis timing metrics"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional context like raw tool calls"
    )


class StrategyResult(BaseModel):
    """
    Result from strategy execution.
    """

    message: str = Field(description="Status message")
    status: StrategyStatus = Field(description="Execution status")
    should_checkpoint: bool = Field(default=False, description="Whether to save state")
    step_result: Optional[StepResult] = Field(default=None, description="Result of the step")

    @property
    def is_terminal(self) -> bool:
        """
        Checks if status is terminal.
        """

        return self.status in (StrategyStatus.COMPLETE, StrategyStatus.ERROR)


class WorkflowResult(BaseModel):
    """
    Base class for workflow outcomes.
    """

    status: str = Field(default="unknown")
    workflow_id: str = Field(default="", description="Unique ID")
    completion_reason: str = Field(default="", description="Reason for finishing")
    success: bool = Field(default=False, description="Whether workflow achieved goal")

    duration: float = Field(default=0.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    steps_executed: int = Field(default=0)
    error: Optional[str] = Field(default=None)
    step_results: List[StepResult] = Field(default_factory=list)


class IntentResult(WorkflowResult):
    """
    Result of intent workflow.
    """

    intent: str = Field(default="", description="The intent executed")
    steps_taken: int = Field(ge=0, description="Number of steps executed")
    final_screen: Optional[Any] = Field(default=None, description="Final state")

    metrics: Dict[str, Dict[str, float]] = Field(
        default_factory=dict, description="Execution metrics"
    )
    memory_summary: Dict[str, Any] = Field(
        default_factory=dict, description="Summary of Knowledge Graph"
    )


class ExplorationResult(WorkflowResult):
    """
    Result of app exploration.
    """

    unique_screens: int = Field(ge=0, description="Unique screens discovered")
    total_actions: int = Field(ge=0, description="Total actions performed")
    total_transitions: int = Field(ge=0, description="Total transitions")

    discovered_activities: List[str] = Field(default_factory=list)
    coverage_percentage: float = Field(ge=0.0, le=100.0, description="App coverage")

    screen_graph: Dict[str, List[str]] = Field(default_factory=dict)


class ActionResult(BaseModel):
    """
    Result of physical action execution.
    """

    success: bool = Field(description="Execution status")
    duration: int = Field(ge=0, description="Duration in milliseconds")
    error: Optional[str] = Field(default=None, description="Error details")
    output: Optional[str] = Field(default=None, description="Command output")


class ExecutionResult(BaseModel):
    """
    Result of step execution attempt.
    """

    model_config = ConfigDict(frozen=True)

    success: bool = Field(..., description="Whether the execution was successful")
    duration: int = Field(..., description="Duration of execution in milliseconds")

    pre_hash: str = Field(default="", description="Visual hash before execution")
    post_hash: str = Field(default="", description="Visual hash after execution")

    error: Optional[str] = Field(default=None, description="Error message if failed")
    screen_changed: bool = Field(default=False, description="Whether the screen changed")


class PlanResult(BaseModel):
    """
    Result of step planning.
    """

    model_config = ConfigDict(frozen=True)

    reason: str = Field(..., description="Explanation for the plan")
    is_complete: bool = Field(..., description="Whether the intent is achieved")
    step: Optional[Step] = Field(default=None, description="The planned step, if any")

    memories: int = Field(default=0, description="Count of memories used")
    should_retry: bool = Field(default=False, description="Whether to retry analysis")

    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional context")
    metrics: Dict[str, float] = Field(default_factory=dict, description="Performance metrics")

    is_valid_action: bool = Field(default=True, description="Whether the planned action is valid")
    validation_reasoning: Optional[str] = Field(
        default=None, description="Reason if action is invalid"
    )


class GenerateResult(BaseModel):
    """
    Raw result from LLM generation.
    """

    content: str = Field(default="", description="Text content from LLM")
    tool_calls: List[Any] = Field(default_factory=list, description="Structured tool calls")
    metrics: Dict[str, float] = Field(default_factory=dict, description="Token usage and timing")
