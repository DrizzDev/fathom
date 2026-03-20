from __future__ import annotations

from typing import Any, Dict, Sequence, Union

from fathom.core.services.exporter.step_inference import get_condition
from fathom.core.services.exporter.step_record import (
    get_action_type,
    get_conditional_type,
    get_event_type,
    is_explicit_conditional,
)
from fathom.core.services.exporter.step_targets import resolve_target
from fathom.schemas.steps import StepResult


def build_export_payload(
    step_results: Sequence[Union[StepResult, Dict[str, Any]]],
) -> list[Dict[str, Any]]:
    payload: list[Dict[str, Any]] = []
    for index, step in enumerate(step_results, start=1):
        action_type_val = get_action_type(step=step)
        target = resolve_target(step=step)
        condition = get_condition(step=step)
        event_type = get_event_type(step=step)
        is_conditional = is_explicit_conditional(step=step)
        conditional_type = get_conditional_type(step=step)

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
