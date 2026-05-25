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
            logger.debug(
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
    mapping = {
        "scroll": "Scroll down",
        "swipe_up": "Scroll down",
        "swipe_down": "Scroll up",
        "swipe_left": "Swipe left",
        "swipe_right": "Swipe right",
    }
    return mapping.get(action_type, "Scroll")


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
