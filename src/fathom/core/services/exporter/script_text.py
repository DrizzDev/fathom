from __future__ import annotations

import re
from typing import Optional

from fathom.core.services.exporter.constants import (
    DYNAMIC_TARGET_PREFIXES,
    STORE_NAME_PATTERN,
)


def normalize_script_output(script: str) -> str:
    cleaned_lines = [line.rstrip() for line in str(script).replace("\r\n", "\n").split("\n")]
    while cleaned_lines and not cleaned_lines[-1].strip():
        cleaned_lines.pop()
    if not cleaned_lines:
        return ""
    return "\n".join(cleaned_lines) + "\n"


def normalize_text_signal(text: str) -> str:
    cleaned = re.sub(pattern=r"[^a-z0-9\s]", repl=" ", string=str(text).lower())
    return re.sub(pattern=r"\s+", repl=" ", string=cleaned).strip()


def intent_mentions_phrase(intent: str, phrase: str) -> bool:
    if not intent or not phrase:
        return False

    intent_norm = normalize_text_signal(text=intent)
    phrase_norm = normalize_text_signal(text=phrase)
    return bool(phrase_norm) and phrase_norm in intent_norm


def is_generic_dynamic_reference(text: str) -> bool:
    lower = str(text).strip().lower()
    generic_tokens = (
        "first",
        "second",
        "third",
        "search result",
        "matching result",
        "matching item",
        "selected item",
    )
    return any(token in lower for token in generic_tokens)


def generalize_dynamic_target(target: str, intent: str, *, generic_item_phrase: str) -> str:
    if not target:
        return target

    lowered = target.lower()
    for prefix in DYNAMIC_TARGET_PREFIXES:
        marker = lowered.find(prefix)
        if marker < 0:
            continue

        suffix_start = marker + len(prefix)
        specific_phrase = target[suffix_start:].strip()
        if not specific_phrase:
            return target
        if intent_mentions_phrase(intent=intent, phrase=specific_phrase):
            return target
        if is_generic_dynamic_reference(text=specific_phrase):
            return target

        return f"{target[:suffix_start]}{generic_item_phrase}"

    return target


def sanitize_script_targets(script: str, intent: str) -> str:
    if not script:
        return script

    lines = script.splitlines()
    updated: list[str] = []
    combined_signal = normalize_text_signal(text=f"{intent} {script}")
    search_context = any(
        token in combined_signal
        for token in ("search bar", "search suggestion", "search result", "search")
    )
    generic_item_phrase = "the first search result" if search_context else "the first matching item"

    for line in lines:
        transformed = STORE_NAME_PATTERN.sub(repl="", string=line)
        for prefix in DYNAMIC_TARGET_PREFIXES:
            pattern = re.compile(
                pattern=rf"({re.escape(prefix)})(.+?)(?=(\s+is\s+visible\b|$))",
                flags=re.IGNORECASE,
            )

            def __replace(match: "re.Match[str]") -> str:
                left = match.group(1)
                suffix = match.group(2).strip()
                combined = f"{left}{suffix}"
                generalized = generalize_dynamic_target(
                    target=combined, intent=intent, generic_item_phrase=generic_item_phrase
                )
                if not generalized.lower().startswith(left.lower()):
                    return match.group(0)
                return generalized

            transformed = pattern.sub(repl=__replace, string=transformed)
        transformed = re.sub(pattern=r"\s{2,}", repl=" ", string=transformed).strip()
        updated.append(transformed)

    normalized = "\n".join(updated)
    return normalize_script_output(script=normalized)


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


def intent_requires_if_block(intent: str) -> bool:
    normalized = normalize_text_signal(text=intent)
    if not normalized:
        return False

    conditional_terms = (" if ", " when ", " if_", " if-", " if(", " if the", " if there")
    return any(term in f" {normalized} " for term in conditional_terms)


def extract_target_from_action_line(line: str) -> Optional[str]:
    text = line.strip()
    lower = text.lower()

    if lower.startswith("tap on "):
        return text[len("Tap on ") :].strip()

    if lower.startswith("type "):
        marker = lower.rfind(" into ")
        if marker != -1:
            return text[marker + len(" into ") :].strip()

    if lower.startswith("scroll until you see "):
        return text[len("Scroll until you see ") :].strip()
    if lower.startswith("scroll down until ") or lower.startswith("scroll up until "):
        suffix = (
            text[len("Scroll down until ") :]
            if lower.startswith("scroll down until ")
            else text[len("Scroll up until ") :]
        )
        return suffix.strip()

    if lower.startswith("wait for "):
        return text[len("Wait for ") :].strip()

    if lower.startswith("long press on "):
        return text[len("Long press on ") :].strip()

    if lower.startswith("validate "):
        target = text[len("Validate ") :].strip()
        if target.lower().startswith("that "):
            target = target[5:].strip()
        return target

    if lower.startswith("open_app "):
        return None
    return None
