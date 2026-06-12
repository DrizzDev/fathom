from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict, cast

from fathom.constants.state import CommonStateKey, ExplorationStateKey
from fathom.schemas.actions import Action
from fathom.schemas.results import AnalysisResult
from fathom.schemas.screens import ScreenCapture, ScreenState
from fathom.schemas.steps import StepResult


class ExplorationGraphState(TypedDict, total=False):
    """
    State flowing through the exploration graph; strictly typed and serialisable.

    The keys mirror the CommonStateKey/ExplorationStateKey values so the nodes
    can read and write them through those enums; declaring them here registers
    each as a LangGraph channel that persists across nodes.
    """

    CAPTURE: Optional[ScreenCapture]
    SCREEN_STATE: Optional[ScreenState]
    IS_NEW_SCREEN: bool

    ANALYSIS: Optional[AnalysisResult]
    ACTION: Optional[Action]
    CONTENT_EXHAUSTED: bool

    STEP_RESULT: Optional[StepResult]
    STEP_RESULTS: List[StepResult]

    BFS_PHASE: str

    IS_COMPLETE: bool
    COMPLETION_REASON: Optional[str]
    STEP_NUMBER: int

    GROUNDING_DURATION: float
    ANALYSIS_DURATION: float
    EXECUTION_DURATION: float


def __values(state: ExplorationGraphState) -> Dict[str, Any]:
    """
    Views the typed state as a plain mapping for enum-keyed access.
    """

    return cast("Dict[str, Any]", state)


def get_capture(state: ExplorationGraphState) -> Optional[ScreenCapture]:
    """
    Get the captured screen from state, if present.
    """

    return __values(state).get(CommonStateKey.CAPTURE)


def get_screen_state(state: ExplorationGraphState) -> Optional[ScreenState]:
    """
    Get the computed screen state, if present.
    """

    return __values(state).get(CommonStateKey.SCREEN_STATE)


def get_action(state: ExplorationGraphState) -> Optional[Action]:
    """
    Get the action chosen for this step, if any.
    """

    return __values(state).get(ExplorationStateKey.ACTION)


def get_step_result(state: ExplorationGraphState) -> Optional[StepResult]:
    """
    Get the result of the executed step, if any.
    """

    return __values(state).get(CommonStateKey.STEP_RESULT)


def get_step_results(state: ExplorationGraphState) -> List[StepResult]:
    """
    Get the accumulated step results.
    """

    return cast("List[StepResult]", __values(state).get(ExplorationStateKey.STEP_RESULTS, []))


def get_bfs_phase(state: ExplorationGraphState, default: str) -> str:
    """
    Get the published DFS phase, falling back to a default.
    """

    return cast("str", __values(state).get(ExplorationStateKey.BFS_PHASE, default))


def is_complete(state: ExplorationGraphState) -> bool:
    """
    Whether the run has been marked complete.
    """

    return bool(__values(state).get(CommonStateKey.IS_COMPLETE, False))


def is_content_exhausted(state: ExplorationGraphState) -> bool:
    """
    Whether the current screen has been fully scanned.
    """

    return bool(__values(state).get(ExplorationStateKey.CONTENT_EXHAUSTED, False))
