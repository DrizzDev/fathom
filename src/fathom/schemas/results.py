from __future__ import annotations

from enum import StrEnum
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field

from fathom.constants import StrategyStatus
from fathom.constants.exploration import FocusRelevance
from fathom.constants.screen import ScreenCategory
from fathom.schemas.actions import Action
from fathom.schemas.artifacts import ScreenArtifact
from fathom.schemas.content import ScreenContent
from fathom.schemas.delta import DeltaSignal
from fathom.schemas.screens import ScreenCapture
from fathom.schemas.steps import Step, StepResult
from fathom.schemas.swipe import SwipeExecution


class AnalysisOutcome(StrEnum):
    """
    What the agent decided to do this ANALYZE turn.

    Values route through distinct planner paths:

    - ``ACT``: the agent committed to a concrete action; ``action`` is load-bearing and EXECUTE consumes it.
    - ``ASK_USER``: the agent wants the human to clarify the next move (existing HITL path; planner emits an ``ASK_USER`` action).
    """

    ACT = "act"
    ASK_USER = "ask_user"


class AnalysisResult(BaseModel):
    """
    Result of vision analysis.
    """

    action: Action = Field(
        description="Primary recommended action.",
    )
    alternatives: List[Action] = Field(
        default_factory=list, description="Alternative actions considered"
    )
    reasoning: str = Field(description="Reasoning process")
    screen_description: str = Field(description="Description of screen content")
    is_goal_complete: bool = Field(
        default=False, description="Whether the user intent has been fully achieved"
    )
    goal_completion_reason: Optional[str] = Field(
        default=None,
        description="Explicit reason why the goal is complete (e.g., 'Order placed successfully', 'Feature verified on screen'). Used for intent verification.",
    )
    is_sub_goal_complete: bool = Field(
        default=False,
        description="Whether the current decomposed sub-goal is complete",
    )
    subgoal_completion_reason: Optional[str] = Field(
        default=None,
        description="Explicit reason why the sub-goal is complete (e.g., 'Item added to cart', 'User authenticated'). Used for verification and audit trails.",
    )
    completion_criteria_met: Optional[List[str]] = Field(
        default=None,
        description="List of criteria/conditions that triggered completion (e.g., ['payment_processed', 'order_confirmed']). For multi-condition verifications.",
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
    content_exhausted: bool = Field(
        default=False, description="Model signals end of scrollable content"
    )
    delta: Optional[DeltaSignal] = Field(
        default=None, description="Optional model-provided semantic delta hints"
    )
    focus_relevance: Optional[FocusRelevance] = Field(
        default=None,
        description="How this screen relates to the exploration focus; None when the model did not classify.",
    )
    category: Optional[ScreenCategory] = Field(
        default=None,
        description="The functional kind of screen describe_screen classified; None when unclassified.",
    )
    content: Optional[ScreenContent] = Field(
        default=None,
        description="Structured screen content (purpose, elements, actions); None when not described.",
    )

    outcome: AnalysisOutcome = Field(
        default=AnalysisOutcome.ACT,
        description=(
            "What the agent decided this turn. ``ACT`` is the default and "
            "consumes ``action``. ``ASK_USER`` routes through HITL without "
            "inventing a synthetic UI action."
        ),
    )


class ToolErrorFeedback(BaseModel):
    """
    Structured feedback about a failed tool invocation that can be shown to the model.
    """

    tool_name: str = Field(description="Name of the tool that failed")

    tool_call_id: Optional[str] = Field(
        default=None,
        description=(
            "Optional identifier correlating this error to the originating tool call "
            "(if provided by the LLM adapter)."
        ),
    )
    error_kind: Literal["validation", "execution"] = Field(
        description="Whether the failure happened during validation or execution"
    )
    message: str = Field(
        description=(
            "Concise, model-ready description of what went wrong and how to fix it "
            "(e.g. missing fields, wrong types, or device/runtime failure)."
        )
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

    executed_subgoals: List[str] = Field(
        default_factory=list,
        description="List of sub-goal descriptions that were executed and completed",
    )
    skipped_subgoals: List[str] = Field(
        default_factory=list,
        description="List of sub-goal descriptions that were skipped (should be empty for successful execution)",
    )
    subgoal_count: int = Field(
        default=0, ge=0, description="Total number of sub-goals in the decomposition"
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

    screen_graph: Dict[str, Any] = Field(default_factory=dict)


class ActionResult(BaseModel):
    """
    Result of physical action execution.
    """

    success: bool = Field(description="Execution status")
    duration: int = Field(ge=0, description="Duration in milliseconds")
    error: Optional[str] = Field(default=None, description="Error details")
    output: Optional[str] = Field(default=None, description="Command output")


class ActionTraceAttempt(BaseModel):
    """
    Attempt metadata for one dispatched trace event.
    """

    model_config = ConfigDict(frozen=True)

    index: int = Field(ge=0, description="Zero-based attempt index within one logical step.")


class ActionTraceEvent(BaseModel):
    """
    One concrete action trace captured during execution.
    """

    model_config = ConfigDict(frozen=True)

    capture: ScreenCapture = Field(
        description="Pre-action capture the trace should be rendered on.",
    )
    coords: Tuple[int, ...] = Field(
        description="Action coordinates to render for this trace event.",
    )
    attempt: Optional[ActionTraceAttempt] = Field(
        default=None,
        description="Attempt metadata when the action dispatched multiple device commands.",
    )


class TraceEmission(BaseModel):
    """
    Adapter-layer outcome of staging one rendered trace through the artifact pipeline.
    Composes the source gesture event with an optional artifact handle so future metadata extends here.
    """

    model_config = ConfigDict(frozen=True)

    event: ActionTraceEvent = Field(
        description="Source gesture event the trace was rendered from.",
    )
    artifact: Optional[ScreenArtifact] = Field(
        default=None,
        description="Pipeline-staged trace artifact when emission succeeded; None when un-wired or skipped.",
    )


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
    is_cancelled: bool = Field(default=False, description="Whether the execution was cancelled")
    swipe_execution: Optional[SwipeExecution] = Field(
        default=None,
        description="Bounded swipe execution outcome (attempts, rejections, abort reason) when available.",
    )
    trace_emissions: Tuple[TraceEmission, ...] = Field(
        default_factory=tuple,
        description="Trace emissions captured during execution; each wraps the gesture event and its staged artifact handle.",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional finalization markers such as partial completion or the phase that timed out.",
    )


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
