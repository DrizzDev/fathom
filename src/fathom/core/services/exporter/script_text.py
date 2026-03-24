from __future__ import annotations

from typing import Optional

from fathom.core.services.exporter.constants import EXECUTABLE_ACTION_PREFIXES

# Map prefix → action kind. Most are the prefix without the trailing space;
# "swipe " is the exception and maps to "scroll".
_PREFIX_TO_KIND = {
    prefix: "scroll" if prefix == "swipe " else prefix.strip().replace(" ", "_")
    for prefix in EXECUTABLE_ACTION_PREFIXES
}


def normalize_script_output(script: str) -> str:
    cleaned_lines = [line.rstrip() for line in str(script).replace("\r\n", "\n").split("\n")]
    while cleaned_lines and not cleaned_lines[-1].strip():
        cleaned_lines.pop()
    if not cleaned_lines:
        return ""
    return "\n".join(cleaned_lines) + "\n"


def is_structural_script_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if stripped == "{" or stripped == "}":
        return True
    return stripped.lower().startswith("if ")


def action_kind_from_line(line: str) -> Optional[str]:
    normalized = line.strip().lower()
    if not normalized:
        return None

    for prefix, kind in _PREFIX_TO_KIND.items():
        if normalized.startswith(prefix):
            return kind
    return None
