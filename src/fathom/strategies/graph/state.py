from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict

from fathom.schemas.execution import ExecutionContext
from fathom.schemas.observation import ScreenObservation
from fathom.schemas.outcomes import ActionOutcome
from fathom.schemas.results import AnalysisResult, PlanResult
from fathom.schemas.screens import ScreenCapture, ScreenState
from fathom.schemas.steps import Step, StepResult


class IntentGraphState(TypedDict, total=False):
    """
    State flowing through the Intent Execution Graph. Strictly typed and serializable.
    """

    INTENT: str
    USE_XML: bool
    MAX_STEPS: int
    STEP_NUMBER: int
    IS_COMPLETE: bool
    SHOULD_RETRY: bool
    INJECTED_CONTEXT: Optional[str]
    COMPLETION_REASON: Optional[str]

    IS_NEW_SCREEN: bool
    CAPTURE: Optional[ScreenCapture]
    SCREEN_STATE: Optional[ScreenState]
    SCREEN_OBSERVATION: Optional[ScreenObservation]

    XML_CONTENT: Optional[str]
    ELEMENTS: Optional[Dict[str, Any]]

    PLAN: Optional[PlanResult]
    PLANNED_STEP: Optional[Step]
    STEP_RESULTS: List[StepResult]
    STEP_RESULT: Optional[StepResult]
    ANALYSIS: Optional[AnalysisResult]

    # Coordination handoff: SUPERVISE writes, EXECUTE / OBSERVE read.
    # Must be declared on the schema or LangGraph drops the channel update.
    EXECUTION_CONTEXT: Optional[ExecutionContext]
    EXECUTION_BLOCKED: bool

    # Supervisor feedback channel into the next planner turn. Populated
    # by SUPERVISE on a block, cleared on a successful allow path.
    LAST_BLOCK_REASON: Optional[str]
    LAST_BLOCK_MESSAGE: Optional[str]

    # OBSERVE writes the post-action outcome; RECORD reads it.
    ACTION_OUTCOME: Optional[ActionOutcome]

    ANALYSIS_DURATION: float
    EXECUTION_DURATION: float
    GROUNDING_DURATION: float

    # Post-action activity captured in EXECUTE, consumed in RECORD
    POST_ACTIVITY: Optional[str]

    # Sub-goal state (for global propagation across graph nodes)
    CURRENT_SUB_GOAL_INDEX: int
    AGENT_STATE_CHECKPOINT: Optional[Dict[str, object]]
