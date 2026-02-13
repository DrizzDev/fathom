"""
LangGraph state for the BFS exploration graph.

Parallel to :class:`FathomGraphState` but tailored for the BFS
exploration workflow.  Heavy mutable objects (BFS queue, knowledge
graph, services) live on the :class:`ExplorationNodeContext` closure,
not in this dict.
"""

from __future__ import annotations

from typing import List, Optional

from typing_extensions import TypedDict

from fathom.schemas.actions import Action
from fathom.schemas.results import AnalysisResult
from fathom.schemas.screens import ScreenCapture, ScreenState
from fathom.schemas.steps import StepResult


class ExplorationGraphState(TypedDict, total=False):
    """
    LangGraph state flowing through the exploration execution graph.

    Design decisions mirror :class:`FathomGraphState`:
    - ``total=False`` so nodes only set the fields they produce.
    - BFS mutable state (queue, paths, sets) lives on the context closure.
    - Only serializable, lightweight values flow through the graph dict.
    """

    # ── Configuration (set once at graph start) ────────────────────────
    max_steps: int

    # ── Per-step mutable state ─────────────────────────────────────────
    step_number: int
    capture: Optional[ScreenCapture]
    screen_state: Optional[ScreenState]
    is_new_screen: bool

    # ── BFS phase routing ──────────────────────────────────────────────
    bfs_phase: str  # "scan", "return", "advance"

    # ── Scan / VLM artefacts ───────────────────────────────────────────
    action: Optional[Action]
    analysis: Optional[AnalysisResult]
    kg_context: str
    screen_description: Optional[str]
    content_exhausted: bool
    analysis_duration: float

    # ── Execution artefacts ────────────────────────────────────────────
    step_result: Optional[StepResult]

    # ── Accumulated history ────────────────────────────────────────────
    step_results: List[StepResult]

    # ── Terminal conditions ────────────────────────────────────────────
    is_complete: bool
    completion_reason: Optional[str]

    # ── Timing (for UX / metrics) ─────────────────────────────────────
    grounding_duration: float
    execution_duration: float
