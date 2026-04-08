"""Provider-neutral trace-entry formatting helpers.

Trace entries arrive in two flavors throughout the codebase: as plain
dicts (when persisted from a remote run, replayed from disk, or built
synthetically by tests) and as ``Action``-like objects with attributes.
The helpers here normalize both shapes so adapter renderers and
verification prompts share a single source of truth.
"""

from __future__ import annotations

from typing import Any, Mapping, Tuple

from fathom.schemas.actions import resolve_action_target

__all__ = ["extract_action_fields", "format_trace_action_line"]


def _read(action: Any, field: str, default: Any = None) -> Any:
    """Uniformly read a field from dict- or object-shaped actions."""

    if isinstance(action, dict):
        return action.get(field, default)
    return getattr(action, field, default)


def extract_action_fields(entry: Mapping[str, Any]) -> Tuple[str, str]:
    """Return ``(action_type, target)`` from a trace entry.

    Handles dict-shaped action payloads (``entry["action"] = {...}``)
    and object-shaped ones (``entry["action"]`` is an ``Action``
    instance). Enum-typed ``action_type`` values are unwrapped to
    their string value.

    The actual target-routing decision is delegated to
    :func:`fathom.schemas.actions.resolve_action_target` so this
    helper, ``trace_payload._resolve_target``, and
    ``Action.to_description`` all share one canonical chain: the
    per-kind subject (validate → ``validation_subject``, wait →
    ``wait_subject``, swipe/scroll → ``scroll_target``), then
    ``target_name``/``target`` → ``export_target`` →
    ``natural_language_target`` → ``label:{id}``, then the
    ``"unknown"`` fallback.
    """

    action = entry.get("action", {})

    action_type = _read(action, "action_type", "unknown")
    if hasattr(action_type, "value") and not isinstance(action_type, str):
        action_type_str = action_type.value
    else:
        action_type_str = str(action_type)

    # Dict-shaped actions from legacy persistence may carry either
    # ``target_name`` (ExecuteAction shape) or ``target`` (Action
    # shape). Try the canonical one first, then fall back.
    primary_target = _read(action, "target_name")
    if primary_target is None:
        primary_target = _read(action, "target")

    resolved = resolve_action_target(
        action_type=action_type_str,
        target_name=primary_target,
        export_target=_read(action, "export_target"),
        natural_language_target=_read(action, "natural_language_target"),
        validation_subject=_read(action, "validation_subject"),
        wait_subject=_read(action, "wait_subject"),
        scroll_target=_read(action, "scroll_target"),
        label_id=_read(action, "label_id"),
    )

    return action_type_str, resolved


def format_trace_action_line(entry: Mapping[str, Any], *, prefix: str = "- ") -> str:
    """Render a trace entry as a one-line ``"<prefix><action_type>: <target>"`` string."""

    action_type, target = extract_action_fields(entry)
    return f"{prefix}{action_type}: {target}"
