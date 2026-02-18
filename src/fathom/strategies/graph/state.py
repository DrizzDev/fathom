from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict

from fathom.schemas.results import AnalysisResult, PlanResult
from fathom.schemas.screens import ScreenCapture, ScreenState
from fathom.schemas.steps import Step, StepResult


class IntentGraphState(TypedDict, total=False):
    """
    State flowing through the Intent Execution Graph. Strictly typed and serializable.
    """

    intent: str
    use_xml: bool
    max_steps: int
    step_number: int
    is_complete: bool
    should_retry: bool
    injected_context: Optional[str]
    completion_reason: Optional[str]

    is_new_screen: bool
    capture: Optional[ScreenCapture]
    screen_state: Optional[ScreenState]

    xml_content: Optional[str]
    elements: Optional[Dict[str, Any]]

    plan: Optional[PlanResult]
    planned_step: Optional[Step]
    step_results: List[StepResult]
    step_result: Optional[StepResult]
    analysis: Optional[AnalysisResult]

    analysis_duration: float
    execution_duration: float
    grounding_duration: float
