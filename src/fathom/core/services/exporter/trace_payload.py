from __future__ import annotations

from typing import Any, Dict, Sequence, Union

from fathom.core.services.exporter.step_record import (
    get_action_type,
    get_activity,
    get_conditional_type,
    get_event_type,
    is_explicit_conditional,
    is_launcher_activity,
)
from fathom.schemas.steps import StepResult


def _get_field(step: Union[StepResult, Dict[str, Any]], field: str, default: Any = None) -> Any:
    """Extract a field from a StepResult or dict, reading from the Action when available."""
    if isinstance(step, StepResult):
        return getattr(step.step.action, field, default)
    return step.get(field, default)


def build_export_payload(
    step_results: Sequence[Union[StepResult, Dict[str, Any]]],
) -> list[Dict[str, Any]]:
    payload: list[Dict[str, Any]] = []
    step_index = 0
    for step in step_results:
        # Skip steps executed on launcher — navigational overhead, not part of the intent.
        if is_launcher_activity(activity=get_activity(step)):
            continue
        step_index += 1
        index = step_index
        action_type_val = get_action_type(step=step)
        event_type = get_event_type(step=step)
        is_conditional = is_explicit_conditional(step=step)
        conditional_type = get_conditional_type(step=step)

        # Use authoritative fields from VLM structured output.
        target = (
            _get_field(step, "export_target")
            or _get_field(step, "natural_language_target")
            or "element"
        )
        condition = _get_field(step, "condition")

        if isinstance(step, StepResult):
            text = step.step.action.text
            rationale = step.step.action.rationale
            screen_changed = bool(step.screen_changed)
        else:
            text = step.get("text")
            rationale = str(object=step.get("rationale") or "")
            screen_changed = bool(step.get("screen_changed", False))

        payload.append(
            {
                "step": index,
                "event_type": event_type,
                "action_type": action_type_val,
                "target": target,
                "text": text,
                "condition": condition,
                "is_conditional": is_conditional,
                "conditional_type": conditional_type,
                "screen_changed": screen_changed,
                "rationale": rationale,
            }
        )
    return payload
