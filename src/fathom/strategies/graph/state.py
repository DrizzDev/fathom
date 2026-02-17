"""
Graph state definitions for LangGraph workflows.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict

from fathom.schemas.results import PlanResult
from fathom.schemas.screens import ScreenCapture, ScreenState
from fathom.schemas.steps import Step, StepResult


class IntentGraphState(TypedDict, total=False):
    """
    State flowing through the Intent Execution Graph.
    Strictly typed and serializable.
    """

    # --- Configuration ---
    intent: str
    max_steps: int
    use_xml: bool

    # --- Execution State ---
    step_number: int
    is_complete: bool
    completion_reason: Optional[str]
    should_retry: bool
    injected_context: Optional[str]

    # --- Artefacts (Per Step) ---
    capture: Optional[ScreenCapture]
    screen_state: Optional[ScreenState]
    is_new_screen: bool

    # Hierarchy processing
    xml_content: Optional[str]
    elements: Optional[Dict[str, Any]]  # Label map

    # Analysis
    plan: Optional[PlanResult]
    planned_step: Optional[Step]

    # Execution
    step_result: Optional[StepResult]

    # --- History ---
    step_results: List[StepResult]

    # --- Metrics ---
    grounding_duration: float
    analysis_duration: float
    execution_duration: float
