from __future__ import annotations

import re
from typing import Optional


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
