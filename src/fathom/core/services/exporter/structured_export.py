from __future__ import annotations

from typing import Dict

from fathom.core.services.exporter.script_text import (
    action_kind_from_line,
    count_action_lines,
    is_structural_script_line,
)


def executable_action_counts(script: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for raw_line in script.splitlines():
        line = raw_line.strip()
        if not line or is_structural_script_line(line):
            continue

        action_kind = action_kind_from_line(line=line)
        if not action_kind:
            continue
        counts[action_kind] = counts.get(action_kind, 0) + 1
    return counts


def is_valid_llm_script(candidate: str, catalog_action_count: int) -> bool:
    if not candidate.strip():
        return False
    if "```" in candidate:
        return False

    balance = 0
    for raw_line in candidate.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        for character in line:
            if character == "{":
                balance += 1
            elif character == "}":
                balance -= 1
                if balance < 0:
                    return False
    if balance != 0:
        return False

    candidate_actions = count_action_lines(script=candidate)
    return not (catalog_action_count > 0 and candidate_actions <= 0)


def last_non_structural_line(script: str) -> str:
    for raw_line in reversed(script.splitlines()):
        line = raw_line.strip()
        if not line:
            continue
        if is_structural_script_line(line):
            continue
        return line
    return ""


def contains_goal_validation(script: str) -> bool:
    last_line = last_non_structural_line(script=script)
    if not last_line:
        return False
    return last_line.lower().startswith("validate")
