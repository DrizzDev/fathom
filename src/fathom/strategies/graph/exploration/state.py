from __future__ import annotations

from typing import Any, Optional

from fathom.constants.state import CommonStateKey, ExplorationStateKey
from fathom.schemas.actions import Action
from fathom.schemas.screens import ScreenCapture, ScreenState
from fathom.schemas.steps import StepResult


# TypedDict-compatible state for LangGraph StateGraph type checking
class ExplorationGraphState(dict[str, Any]):
    """
    State schema flowing through the Exploration Graph.
    Must be dict-based for StateGraph compatibility.
    """

    pass


def get_capture(state: dict[str, Any]) -> Optional[ScreenCapture]:
    """
    Get capture from state.
    """

    return state.get(CommonStateKey.CAPTURE)


def get_screen_state(state: dict[str, Any]) -> Optional[ScreenState]:
    """
    Get screen state from state.
    """

    return state.get(CommonStateKey.SCREEN_STATE)


def get_action(state: dict[str, Any]) -> Optional[Action]:
    """
    Get action from state.
    """

    return state.get(ExplorationStateKey.ACTION)


def get_step_result(state: dict[str, Any]) -> Optional[StepResult]:
    """
    Get step result from state.
    """

    return state.get(CommonStateKey.STEP_RESULT)


def is_complete(state: dict[str, Any]) -> bool:
    """
    Check if execution is complete.
    """

    return bool(state.get(CommonStateKey.IS_COMPLETE, False))


def is_content_exhausted(state: dict[str, Any]) -> bool:
    """
    Check if content is exhausted.
    """

    return bool(state.get(ExplorationStateKey.CONTENT_EXHAUSTED, False))
