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
from fathom.schemas.actions import resolve_action_target
from fathom.schemas.steps import StepResult


def _get_field(step: Union[StepResult, Dict[str, Any]], field: str, default: Any = None) -> Any:
    """Extract a field from a StepResult or dict, reading from the Action when available."""
    if isinstance(step, StepResult):
        return getattr(step.step.action, field, default)
    return step.get(field, default)


def _resolve_target(step: Union[StepResult, Dict[str, Any]], action_type_val: str) -> str:
    """Pick the authoritative human-readable target for the trace payload.

    Thin adapter over :func:`fathom.schemas.actions.resolve_action_target`
    — pulls every candidate field off the ``StepResult`` or dict and
    hands the resolution decision to the single canonical router. The
    router handles per-kind routing (validate → ``validation_subject``,
    wait → ``wait_subject``, swipe/scroll → ``scroll_target``), the
    general chain (``target_name`` → ``export_target`` →
    ``natural_language_target``), the ``label:{id}`` fallback, and
    placeholder rejection in one place.

    The historic ``"element"`` fallback is gone; the router returns
    ``"unknown"`` as a last resort, which downstream consumers treat
    as unresolved via ``is_resolved_target``.
    """

    # StepResult wraps an Action, which exposes ``target`` (not
    # ``target_name``). Dict-shaped steps from legacy persistence may
    # carry either key, so try ``target_name`` first and fall back to
    # ``target``. Either way, the value flows into the resolver's
    # ``target_name`` slot — it is the primary candidate in the
    # general chain regardless of the source field's name.
    primary_target = _get_field(step, "target_name")
    if primary_target is None:
        primary_target = _get_field(step, "target")

    return resolve_action_target(
        action_type=action_type_val,
        target_name=primary_target,
        export_target=_get_field(step, "export_target"),
        natural_language_target=_get_field(step, "natural_language_target"),
        validation_subject=_get_field(step, "validation_subject"),
        wait_subject=_get_field(step, "wait_subject"),
        scroll_target=_get_field(step, "scroll_target"),
        label_id=_get_field(step, "label_id"),
    )


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

        target = _resolve_target(step=step, action_type_val=action_type_val)
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
                # Surface the canonical subject fields so the export LLM
                # can cross-reference them when composing action_validations
                # and IF-block guards.
                "validation_subject": _get_field(step, "validation_subject"),
                "wait_subject": _get_field(step, "wait_subject"),
                "scroll_target": _get_field(step, "scroll_target"),
            }
        )
    return payload
