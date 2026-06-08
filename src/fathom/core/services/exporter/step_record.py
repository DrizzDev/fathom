from __future__ import annotations

from logging import getLogger
from typing import Any, Dict, Literal, Optional, Union, cast

from fathom.constants.execution import LAUNCHER_PACKAGES
from fathom.schemas.steps import StepResult

logger = getLogger(__name__)


def get_event_type(step: Union[StepResult, Dict[str, Any]]) -> str:
    if isinstance(step, StepResult):
        val = getattr(step.step, "event_type", None)
        if not val:
            logger.warning(
                "event_type missing on StepResult step %s; defaulting to 'action'.",
                step.step.step_number,
            )
        return val or "action"

    return str(step.get("event_type", "action") or "action")


def get_action_type(step: Union[StepResult, Dict[str, Any]]) -> str:
    if isinstance(step, StepResult):
        return step.step.action.action_type.value

    return str(step.get("action_type", "unknown"))


def swipe_direction_label(action_type: str) -> str:
    """
    Return the script-line label that preserves the executed gesture direction.

    The mapping must NOT flip vertical direction — the script-replay engine
    interprets ``"Scroll up"`` and ``"Scroll down"`` literally, so any
    inversion here causes the replay to scroll the opposite way and the
    intended target never comes into view. Earlier versions inverted
    ``swipe_up``→``"Scroll down"`` and ``swipe_down``→``"Scroll up"`` on the
    gesture-vs-content convention, which broke replays when the planner had
    in fact chosen the correct direction at execution time.

    The covered keys are exactly the contents of
    :data:`fathom.constants.SWIPE_ACTIONS` — the only set the caller in
    :mod:`action_catalog` ever passes in. A bare ``"scroll"`` (no direction)
    resolves to ``"Scroll up"`` because that is the documented default the
    planner intends when it emits a direction-less scroll.
    """

    mapping = {
        "scroll": "Scroll up",
        "swipe_up": "Scroll up",
        "swipe_down": "Scroll down",
        "swipe_left": "Swipe left",
        "swipe_right": "Swipe right",
    }
    return mapping.get(action_type, "Scroll up")


def get_activity(step: Union[StepResult, Dict[str, Any]]) -> str:
    if isinstance(step, dict):
        # Prefer execution_activity (pre-action screen) for launcher detection;
        # fall back to activity (post-action screen) for general use.
        return str(step.get("execution_activity") or step.get("activity") or "")

    # StepResult: activity is only available if passed via metadata.
    if isinstance(step, StepResult):
        return str(step.step.metadata.get("activity", ""))

    return ""


def is_launcher_activity(activity: str) -> bool:
    text = str(activity or "").strip()
    if not text:
        return False

    package = text.split("/")[0]
    return package in LAUNCHER_PACKAGES


def is_explicit_conditional(step: Union[StepResult, Dict[str, Any]]) -> bool:
    if isinstance(step, StepResult):
        return bool(getattr(step.step.action, "is_conditional", False))

    return bool(step.get("is_conditional", False))


def get_conditional_type(
    step: Union[StepResult, Dict[str, Any]],
) -> Optional[Literal["blocker", "transient", "error", "optional"]]:
    if isinstance(step, StepResult):
        raw = getattr(step.step.action, "conditional_type", None)
    else:
        raw = step.get("conditional_type")

    text = str(raw or "").strip().lower()
    if text in ("blocker", "transient", "error", "optional"):
        return cast("Literal['blocker', 'transient', 'error', 'optional']", text)
    if text:
        logger.warning("Unrecognized conditional_type '%s'; dropping to None.", text)
    return None
