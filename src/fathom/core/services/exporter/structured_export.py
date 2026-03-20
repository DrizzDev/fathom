from __future__ import annotations

import re
from typing import Any, Dict, Optional, Sequence

from fathom.core.services.exporter.script_text import (
    action_kind_from_line,
    count_action_lines,
    is_structural_script_line,
)


def normalize_validation_line(value: Any, *, fallback: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return fallback

    match = re.search(pattern=r"\bvalidate\b.*", string=raw, flags=re.IGNORECASE)
    if match:
        extracted = match.group(0).strip()
        return "Validate" + extracted[len("validate") :] if extracted else fallback

    cleaned = raw.rstrip(".")
    if cleaned.lower().startswith("that "):
        return f"Validate {cleaned}."
    return f"Validate that {cleaned[0].lower() + cleaned[1:] if len(cleaned) > 1 else cleaned.lower()}."


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


def is_valid_llm_script(candidate: str, baseline: str) -> bool:
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

    baseline_actions = count_action_lines(script=baseline)
    candidate_actions = count_action_lines(script=candidate)
    return not (baseline_actions > 0 and candidate_actions <= 0)


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
    action_catalog: Optional[Dict[str, str]] = None,
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

    action_catalog = action_catalog or {}
    order_rank = {action_id: idx for idx, action_id in enumerate(required_action_ids)}
    moved_to_blocks: list[str] = []
    for block in cleaned_blocks:
        condition_lower = str(block.get("condition") or "").strip().lower()
        block_ids = list(block.get("action_ids") or [])
        if "outside the us dropdown" not in condition_lower or not block_ids:
            continue

        last_rank = max(order_rank.get(action_id, -1) for action_id in block_ids)
        candidate_ids: list[str] = []
        for action_id in cleaned_remaining:
            rank = order_rank.get(action_id, -1)
            if rank <= last_rank:
                continue
            action_line = str(action_catalog.get(action_id) or "").strip().lower()
            is_scroll = action_line.startswith("scroll ")
            is_location_selection_tap = action_line.startswith("tap on ") and (
                "washington" in action_line or action_line.endswith(" option")
            )
            if is_scroll or is_location_selection_tap:
                candidate_ids.append(action_id)
                last_rank = rank
                continue
            if candidate_ids:
                break

        if candidate_ids:
            block["action_ids"] = block_ids + candidate_ids
            moved_to_blocks.extend(candidate_ids)

    if moved_to_blocks:
        moved_set = set(moved_to_blocks)
        cleaned_remaining = [
            action_id for action_id in cleaned_remaining if action_id not in moved_set
        ]

    normalized["conditional_blocks"] = cleaned_blocks
    normalized["remaining_action_ids"] = cleaned_remaining
    return normalized
