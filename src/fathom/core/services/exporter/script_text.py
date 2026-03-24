from __future__ import annotations

from typing import Optional


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


def count_action_lines(script: str) -> int:
    count = 0
    for line in script.splitlines():
        if is_structural_script_line(line):
            continue
        count += 1
    return count


def action_kind_from_line(line: str) -> Optional[str]:
    normalized = line.strip().lower()
    if not normalized:
        return None

    if normalized.startswith("open_app "):
        return "open_app"
    if normalized.startswith("tap "):
        return "tap"
    if normalized.startswith("type "):
        return "type"
    if normalized.startswith("scroll "):
        return "scroll"
    if normalized.startswith("swipe "):
        return "scroll"
    if normalized.startswith("wait "):
        return "wait"
    if normalized.startswith("press "):
        return "press"
    if normalized.startswith("long press "):
        return "long_press"
    return None
