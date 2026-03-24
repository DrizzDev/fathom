from __future__ import annotations

from typing import Any, Dict, Sequence

from fathom.core.services.exporter.script_text import (
    action_kind_from_line,
    count_action_lines,
    is_structural_script_line,
)


def normalize_validation_line(value: Any, *, fallback: str) -> str:
    raw = str(value or "").strip()
    return raw if raw else fallback


def normalize_final_validation(value: Any) -> str:
    return normalize_validation_line(
        value=value,
        fallback="Validate expected goal state is visible.",
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


def normalize_structured_action_ids(
    structured_args: Dict[str, Any],
    required_action_ids: Sequence[str],
) -> Dict[str, Any]:
    normalized = dict(structured_args)
    normalized["final_validation"] = normalize_final_validation(
        value=normalized.get("final_validation")
    )
    raw_action_validations = normalized.get("action_validations")
    normalized_action_validations: Dict[str, str] = {}
    if isinstance(raw_action_validations, dict):
        for action_id, validation_text in raw_action_validations.items():
            aid = str(action_id).strip()
            if not aid:
                continue
            normalized_action_validations[aid] = normalize_validation_line(
                value=validation_text,
                fallback="Validate expected state is visible.",
            )
    normalized["action_validations"] = normalized_action_validations
    conditional_blocks_raw = list(normalized.get("conditional_blocks") or [])
    remaining_raw = list(normalized.get("remaining_action_ids") or [])
    required_set = set(required_action_ids)

    seen: set[str] = set()
    cleaned_blocks: list[Dict[str, Any]] = []
    for block in conditional_blocks_raw:
        if not isinstance(block, dict):
            continue
        condition = str(block.get("condition") or "").strip()
        action_ids_raw = block.get("action_ids") or []
        block_ids: list[str] = []
        for action_id in action_ids_raw:
            aid = str(action_id).strip()
            if not aid or aid in seen or aid not in required_set:
                continue
            seen.add(aid)
            block_ids.append(aid)
        cleaned_blocks.append({"condition": condition, "action_ids": block_ids})

    cleaned_remaining: list[str] = []
    for action_id in remaining_raw:
        aid = str(action_id).strip()
        if not aid or aid in seen or aid not in required_set:
            continue
        seen.add(aid)
        cleaned_remaining.append(aid)

    for required_id in required_action_ids:
        if required_id not in seen:
            cleaned_remaining.append(required_id)
            seen.add(required_id)

    normalized["conditional_blocks"] = cleaned_blocks
    normalized["remaining_action_ids"] = cleaned_remaining
    return normalized
