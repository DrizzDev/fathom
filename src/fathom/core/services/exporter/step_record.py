from __future__ import annotations

from typing import Any, Dict, Literal, Optional, Sequence, Union, cast

from fathom.constants.execution import LAUNCHER_PACKAGES
from fathom.schemas.steps import StepResult


def get_event_type(step: Union[StepResult, Dict[str, Any]]) -> str:
    if isinstance(step, StepResult):
        return getattr(step.step, "event_type", "action") or "action"

    return str(object=step.get("event_type", "action") or "action")


def get_action_type(step: Union[StepResult, Dict[str, Any]]) -> str:
    if isinstance(step, StepResult):
        return step.step.action.action_type.value

    return str(object=step.get("action_type", "unknown"))


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
        return str(object=step.get("activity") or "")

    return ""


def is_launcher_activity(activity: str) -> bool:
    text = str(activity or "").strip()
    if not text:
        return False

    package = text.split("/")[0]
    return package in LAUNCHER_PACKAGES


def is_overlay_detected(step: Union[StepResult, Dict[str, Any]]) -> bool:
    if isinstance(step, StepResult):
        return bool(getattr(step.step.action, "overlay_detected", False))

    return bool(step.get("overlay_detected", False))


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
    return None


def default_condition_for_type(
    conditional_type: Optional[Literal["blocker", "transient", "error", "optional"]],
) -> Optional[str]:
    mapping = {
        "blocker": "Blocker prompt is visible",
        "transient": "Transient screen is visible",
        "error": "Error message is displayed",
        "optional": "Optional UI state is visible",
    }
    return mapping.get(conditional_type or "")


def is_generic_wait_condition(condition: Optional[str]) -> bool:
    if not condition:
        return False

    lower = condition.strip().lower()
    generic_wait_phrases = {
        "screen to load is visible",
        "the app is still loading",
        "loading spinner is visible",
    }
    return lower in generic_wait_phrases


def get_raw_condition(step: Union[StepResult, Dict[str, Any]]) -> Optional[str]:
    if isinstance(step, StepResult):
        raw = getattr(step.step, "condition", None) or getattr(step.step.action, "condition", None)
    else:
        raw = step.get("condition")

    text = str(raw).strip() if raw else None
    return text or None


def find_app_launch_boundary(
    steps: Sequence[Union[StepResult, Dict[str, Any]]],
    package_name: str,
) -> int:
    max_launch_steps = 10
    prefix = package_name + "/"

    for j, step in enumerate(iterable=steps):
        if j > max_launch_steps:
            return 0

        activity = get_activity(step=step)
        if activity.startswith(prefix) or activity == package_name:
            return j

    return 0


def infer_open_app_package(
    steps: Sequence[Union[StepResult, Dict[str, Any]]],
    default_package: str,
) -> Optional[str]:
    if default_package:
        return default_package

    if not steps:
        return None

    for step in steps:
        activity = get_activity(step=step)
        if not activity:
            continue
        if "/" in activity:
            return activity.split("/")[0].strip() or None
        return activity.strip() or None

    return None
