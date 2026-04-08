"""Provider-neutral trace-entry formatting helpers.

Trace entries arrive in two flavors throughout the codebase: as plain
dicts (when persisted from a remote run, replayed from disk, or built
synthetically by tests) and as ``Action``-like objects with attributes.
The helpers here normalize both shapes so adapter renderers and
verification prompts share a single source of truth.
"""

from __future__ import annotations

from typing import Any, Mapping, Tuple

from fathom.schemas.actions import GENERIC_TARGET_PLACEHOLDERS

__all__ = ["extract_action_fields", "format_trace_action_line"]


def _is_resolved(value: Any) -> bool:
    if not value:
        return False
    return str(value).strip().lower() not in GENERIC_TARGET_PLACEHOLDERS


def _read(action: Any, field: str, default: Any = None) -> Any:
    """Uniformly read a field from dict- or object-shaped actions."""

    if isinstance(action, dict):
        return action.get(field, default)
    return getattr(action, field, default)


def extract_action_fields(entry: Mapping[str, Any]) -> Tuple[str, str]:
    """Return (action_type, target) from a trace entry.

    Handles dict-shaped action payloads (``entry["action"] = {...}``) and
    object-shaped ones (``entry["action"]`` is an ``Action`` instance).
    Enum-typed action_type values are unwrapped to their string value.

    Routes the rendered target per action kind so the LLM-facing trace
    history shows the canonical subject instead of the generic
    ``"element"`` placeholder:

    * ``validate`` → ``validation_subject``
    * ``wait``     → ``wait_subject``
    * ``scroll`` / ``swipe_*`` → ``scroll_target``
    * everything else → ``target`` → ``export_target`` → ``natural_language_target``

    Placeholder strings ("element", "button", ...) are treated as
    unresolved so parsing-time fallbacks don't leak into the prompt.
    Missing fields fall back to ``"unknown"``.
    """

    action = entry.get("action", {})

    action_type = _read(action, "action_type", "unknown")
    if hasattr(action_type, "value") and not isinstance(action_type, str):
        action_type_str = action_type.value
    else:
        action_type_str = str(action_type)

    kind = action_type_str.lower()

    candidates: list[Any] = []
    if kind == "validate":
        candidates.append(_read(action, "validation_subject"))
    elif kind == "wait":
        candidates.append(_read(action, "wait_subject"))
    elif "swipe" in kind or kind == "scroll":
        candidates.append(_read(action, "scroll_target"))

    candidates.extend(
        [
            _read(action, "target"),
            _read(action, "export_target"),
            _read(action, "natural_language_target"),
        ]
    )

    for candidate in candidates:
        if _is_resolved(candidate):
            return action_type_str, str(candidate)

    # Nothing resolved — emit the raw target so the caller can see why.
    raw_target = _read(action, "target", "unknown")
    return action_type_str, str(raw_target) if raw_target is not None else "unknown"


def format_trace_action_line(entry: Mapping[str, Any], *, prefix: str = "- ") -> str:
    """Render a trace entry as a one-line ``"<prefix><action_type>: <target>"`` string."""

    action_type, target = extract_action_fields(entry)
    return f"{prefix}{action_type}: {target}"
