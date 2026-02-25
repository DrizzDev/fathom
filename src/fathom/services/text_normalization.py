from __future__ import annotations

import re
from typing import Optional

_MULTISPACE_RE = re.compile(r"\s+")


def clean_text(text: Optional[str]) -> str:
    """Normalize whitespace and trim text."""
    if not text:
        return ""
    cleaned = _MULTISPACE_RE.sub(" ", str(text)).strip()
    cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
    return cleaned


def sentence_case(text: Optional[str]) -> str:
    """Apply lightweight sentence-case normalization."""
    cleaned = clean_text(text)
    if not cleaned:
        return ""
    if cleaned[0].isalpha():
        cleaned = cleaned[0].upper() + cleaned[1:]
    return cleaned


def normalize_reasoning(text: Optional[str]) -> str:
    """Normalize freeform reasoning/evidence text without changing semantics."""
    return sentence_case(text)


def normalize_wait_condition(
    condition: Optional[str], rationale: Optional[str] = None
) -> Optional[str]:
    """Normalize wait condition wording into clear, deterministic phrases."""
    cleaned = clean_text(condition)
    if not cleaned:
        return condition

    lower = cleaned.lower()
    rationale_lower = clean_text(rationale).lower()

    if "app to finish loading" in lower or "splash" in lower:
        return "the app is still loading"
    if "first search result" in lower or "search result" in lower:
        return "search results are still loading"
    if "loading indicator" in lower:
        if "search" in rationale_lower or "result" in rationale_lower:
            return "search results are still loading"
        return "content is still loading"
    return cleaned


def describe_action(action_type: str, target: str, text: Optional[str] = None) -> str:
    """Build canonical action descriptions with stable grammar."""
    kind = clean_text(action_type).lower()
    cleaned_target = clean_text(target) or "UI Element"

    if kind == "tap":
        return f"Tap on {cleaned_target}"
    if kind == "type":
        return f"Type '{clean_text(text)}' into {cleaned_target}"
    if "swipe" in kind:
        direction = kind.split("_")[-1] if "_" in kind else "content"
        return f"Swipe {direction} on {cleaned_target}"
    if kind in ("back", "press_back"):
        return "Press back button"
    if kind in ("home", "press_home"):
        return "Press home button"
    if kind == "enter":
        return "Press enter"
    if kind == "wait":
        if cleaned_target.lower() == "app to finish loading":
            return "Wait for the app to finish loading"
        return f"Wait for {cleaned_target}"
    if kind == "validate":
        return f"Validate {cleaned_target}"
    if kind == "scroll":
        return f"Scroll until you see {cleaned_target}"
    if kind == "long_press":
        return f"Long press on {cleaned_target}"
    if kind == "complete":
        return f"Validate {cleaned_target} (Goal complete)"
    return f"{kind.replace('_', ' ').capitalize()} on {cleaned_target}"


def describe_validation(target: str, *, explicit: bool = False, complete: bool = False) -> str:
    """Build explicit validation descriptions."""
    cleaned_target = clean_text(target) or "Goal State"
    if complete:
        return f"Validate {cleaned_target} (Goal complete)"
    if explicit:
        return f"Validate that {cleaned_target}"
    return f"Validate {cleaned_target}"
