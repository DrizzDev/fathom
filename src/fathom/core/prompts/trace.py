"""Provider-neutral trace-entry formatting helpers.

Trace entries arrive in two flavors throughout the codebase: as plain
dicts (when persisted from a remote run, replayed from disk, or built
synthetically by tests) and as ``Action``-like objects with attributes.
The helpers here normalize both shapes so adapter renderers and
verification prompts share a single source of truth.
"""

from __future__ import annotations

from typing import Any, Mapping, Tuple

__all__ = ["extract_action_fields", "format_trace_action_line"]


def extract_action_fields(entry: Mapping[str, Any]) -> Tuple[str, str]:
    """Return (action_type, target) from a trace entry.

    Handles dict-shaped action payloads (``entry["action"] = {...}``) and
    object-shaped ones (``entry["action"]`` is an ``Action`` instance).
    Enum-typed action_type values are unwrapped to their string value.
    Missing fields fall back to ``"unknown"`` so callers never have to
    guard against ``None``.
    """

    action = entry.get("action", {})
    if isinstance(action, dict):
        target = action.get("target", "unknown")
        action_type = action.get("action_type", "unknown")
    else:
        target = getattr(action, "target", "unknown")
        action_type = getattr(action, "action_type", "unknown")

    if hasattr(action_type, "value") and not isinstance(action_type, str):
        action_type_str = action_type.value
    else:
        action_type_str = str(action_type)

    return action_type_str, str(target)


def format_trace_action_line(entry: Mapping[str, Any], *, prefix: str = "- ") -> str:
    """Render a trace entry as a one-line ``"<prefix><action_type>: <target>"`` string."""

    action_type, target = extract_action_fields(entry)
    return f"{prefix}{action_type}: {target}"
