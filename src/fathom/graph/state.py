from __future__ import annotations

from typing import Any, Dict, List, Optional

from typing_extensions import TypedDict

from fathom.schemas.results import AnalysisResult, PlanResult
from fathom.schemas.screens import ScreenCapture, ScreenState
from fathom.schemas.steps import Step, StepResult


class FathomGraphState(TypedDict, total=False):
    """
    LangGraph state flowing through the intent execution graph.

    Design decisions:
    - ``total=False`` so nodes only need to set the fields they produce.
    - Heavy mutable objects (``AgentState``, services) live on the node
      closures, NOT in this dict — they contain deques/sets that don't
      serialize cleanly for LangGraph checkpointing.
    - Every field maps 1:1 to an existing Fathom schema; no duplication.
    """

    # ── Configuration (set once at graph start) ────────────────────────
    intent: str
    max_steps: int
    use_xml: bool

    # ── Per-step mutable state ─────────────────────────────────────────
    step_number: int
    capture: Optional[ScreenCapture]
    screen_state: Optional[ScreenState]
    is_new_screen: bool

    # XML hierarchy artefacts
    planning_screen: Optional[ScreenCapture]
    elements: Dict[str, Any]

    # LLM analysis artefacts
    plan: Optional[PlanResult]
    analysis: Optional[AnalysisResult]
    planned_step: Optional[Step]
    knowledge: Dict[str, Any]
    analysis_duration: float

    # Execution artefacts
    step_result: Optional[StepResult]

    # ── Accumulated history ────────────────────────────────────────────
    step_results: List[StepResult]

    # ── Terminal conditions ────────────────────────────────────────────
    is_complete: bool
    should_retry: bool
    completion_reason: Optional[str]

    # ── Timing (for UX / metrics) ─────────────────────────────────────
    grounding_duration: float
    hierarchy_duration: float
    execution_duration: float
