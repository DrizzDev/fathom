from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, TypedDict

from fathom.schemas.actions import Action
from fathom.schemas.results import AnalysisResult
from fathom.schemas.screens import ScreenCapture, ScreenState
from fathom.schemas.steps import StepResult


class ExplorationGraphState(TypedDict, total=False):
    """
    State flowing through the Exploration Graph.
    """

    # --- Configuration ---
    max_steps: int

    # --- Execution State ---
    step_number: int
    is_complete: bool
    completion_reason: Optional[str]

    # BFS State
    bfs_phase: str  # "scan", "return", "advance"
    bfs_queue: List[Dict[str, Any]]  # serialized BFSQueueEntry
    visited_hashes: List[str]
    current_path: List[Tuple[str, Dict[str, Any]]]  # (hash, serialized_action)
    pending_nav: List[Dict[str, Any]]  # serialized Action
    scanning_hash: Optional[str]
    root_hash: Optional[str]

    # --- Artefacts ---
    capture: Optional[ScreenCapture]
    screen_state: Optional[ScreenState]
    is_new_screen: bool

    # Scan artefacts
    action: Optional[Action]
    analysis: Optional[AnalysisResult]
    content_exhausted: bool

    # Execution
    step_result: Optional[StepResult]

    # History
    step_results: List[StepResult]

    # Metrics
    grounding_duration: float
    execution_duration: float
    analysis_duration: float
