from __future__ import annotations

import re
from typing import Any, Dict, Optional, Union

from fathom.core.services.exporter.constants import GENERIC_TARGETS, NUMERIC_ORDINAL_RE, ORDINAL_MAP
from fathom.core.services.exporter.step_record import get_action_type
from fathom.core.services.normalizer import Normalizer
from fathom.schemas.steps import StepResult


def is_likely_launch_tap(target: str, description: str) -> bool:
    combined = f"{target} {description}".strip().lower()
    if not combined:
        return False

    if any(
        phrase in combined
        for phrase in [
            "app icon",
            "launcher icon",
            "home screen",
            "launcher button",
        ]
    ):
        return True

    if re.search(r"\b(?:the\s+)?[a-z0-9.\-_'\s]+\s+icon\b", combined):
        return True

    return combined.endswith(" icon")


def normalize_positional(target: str) -> str:
    if not target:
        return target

    text = target.strip()

    def __replace_numeric(match: "re.Match[str]") -> str:
        full = match.group(0).lower()
        return ORDINAL_MAP.get(full, full)

    normalized = NUMERIC_ORDINAL_RE.sub(repl=__replace_numeric, string=text)

    word_ordinals = (
        "first",
        "second",
        "third",
        "fourth",
        "fifth",
        "sixth",
        "seventh",
        "eighth",
        "ninth",
        "tenth",
    )
    stripped = (
        re.sub(pattern=r"^(?:the|a|an)\s+", repl="", string=normalized, flags=re.IGNORECASE)
        .strip()
        .lower()
    )
    is_positional = any(stripped.startswith(o) for o in word_ordinals)

    if not is_positional:
        return target

    without_article = re.sub(
        pattern=r"^(?:the|a|an)\s+", repl="", string=normalized, flags=re.IGNORECASE
    ).strip()
    return f"the {without_article}"


def is_intent_target(target: str, intent: str) -> bool:
    if not target or not intent:
        return False

    target_lower = target.lower()
    intent_lower = intent.lower()

    if target_lower in intent_lower:
        return True

    target_words = set(target_lower.replace("_", " ").split())
    filler = {
        "the",
        "a",
        "an",
        "on",
        "in",
        "to",
        "of",
        "is",
        "and",
        "or",
        "item",
        "button",
        "icon",
        "area",
        "field",
        "for",
        "with",
        "from",
        "by",
        "at",
    }
    meaningful = target_words - filler
    if not meaningful:
        return True

    intent_words = set(intent_lower.replace("_", " ").split())
    overlap = meaningful & intent_words
    return len(overlap) >= len(meaningful) * 0.5


def extract_goal_label(goal_state: str) -> str:
    from fathom.core.services.exporter.constants import LABEL_STOP, SCREEN_RE

    if not goal_state:
        return ""

    trimmed = goal_state.strip().rstrip(".")
    if len(trimmed) <= 60 and "." not in trimmed:
        return trimmed

    matches = SCREEN_RE.findall(string=goal_state)

    for name, kind in reversed(matches):
        cleaned = name.strip()
        words = cleaned.lower().split()

        if len(cleaned) > 1 and not any(w in LABEL_STOP for w in words):
            return f"{cleaned.title()} {kind.strip().title()}"

    return ""


def infer_target_from_rationale(
    *, action_type: str, rationale: Optional[str], fallback: str
) -> str:
    if action_type != "tap":
        return fallback

    raw = str(rationale or "").strip()
    if not raw:
        return fallback

    text = Normalizer.clean(text=raw)
    if not text:
        return fallback

    quoted_match = re.search(
        pattern=r"['\"]([^'\"]{2,80})['\"]\s*(button|tab|icon|option|field|selector|link)?",
        string=raw,
        flags=re.IGNORECASE,
    )
    if quoted_match:
        name = Normalizer.clean(text=quoted_match.group(1))
        suffix = Normalizer.clean(text=quoted_match.group(2) or "")
        if name and name.lower() not in GENERIC_TARGETS:
            return f"{name} {suffix}".strip()

    intent_match = re.search(
        pattern=(
            r"(?:tap|click|press|find|locate|search\s+for|look\s+for)"
            r"\s+(?:on\s+)?(?:the\s+)?"
            r"([a-z0-9][a-z0-9\s&/()+._-]{2,80}?)"
            r"\s*(button|tab|icon|option|field|selector|link)?"
            r"(?:\s+to\b|\.|,|;|$)"
        ),
        string=text,
        flags=re.IGNORECASE,
    )
    if intent_match:
        name = Normalizer.clean(text=intent_match.group(1))
        suffix = Normalizer.clean(text=intent_match.group(2) or "")
        candidate = f"{name} {suffix}".strip()
        lowered = candidate.lower()
        if (
            candidate
            and lowered not in GENERIC_TARGETS
            and lowered not in ("app", "application", "screen")
        ):
            return candidate

    return fallback


def should_generalize_target(rationale: Optional[str]) -> bool:
    if not rationale:
        return False

    text = str(rationale).lower()

    pattern = r"\b(first|any|random|any available|available)\s+(item|product|option|category|choice|element)\b"
    return bool(re.search(pattern=pattern, string=text, flags=re.IGNORECASE))


def generalize_product_target(target: str, rationale: Optional[str]) -> str:
    if not target:
        return target

    cleaned = Normalizer.clean(text=target)
    if not cleaned:
        return target

    rationale_lower = str(rationale).lower() if rationale else ""
    detected_element_type = None

    if "button" in rationale_lower:
        detected_element_type = "button"
    elif "icon" in rationale_lower:
        detected_element_type = "icon"
    elif "option" in rationale_lower:
        detected_element_type = "option"

    button_for_match = re.search(
        pattern=r"^([A-Z][A-Z\s]+|[A-Z][a-z]+(?:\s+[a-z]+)?)?\s*(button|icon|option)\s+for\s+.+$",
        string=cleaned,
        flags=re.IGNORECASE,
    )
    if button_for_match:
        action = Normalizer.clean(text=button_for_match.group(1) or "")
        element_type = Normalizer.clean(
            text=button_for_match.group(2) or detected_element_type or "button"
        )
        if action:
            return f"{action} {element_type}".strip()
        else:
            return f"{element_type}".strip()

    return target


def resolve_target(step: Union[StepResult, Dict[str, Any]]) -> str:
    rationale: Optional[str]
    script_target: Optional[str] = None
    scroll_target: Optional[str] = None
    wait_subject: Optional[str] = None

    if isinstance(step, StepResult):
        if step.generalized_target:
            return normalize_positional(target=step.generalized_target)
        action = step.step.action
        target = action.natural_language_target or action.target
        rationale = action.rationale
        script_target = action.script_target
        scroll_target = action.scroll_target
        wait_subject = action.wait_subject
    else:
        if step.get("generalized_target"):
            raw = str(object=step.get("generalized_target") or "")
            return normalize_positional(target=raw)
        target = step.get("natural_language_target") or step.get("target") or ""
        rationale = str(object=step.get("rationale") or "")
        script_target = step.get("script_target")
        scroll_target = step.get("scroll_target")
        wait_subject = step.get("wait_subject")

    resolved_target = Normalizer.clean(text=target) or "element"
    lower_resolved = resolved_target.lower()

    if lower_resolved in GENERIC_TARGETS:
        for candidate in (script_target, scroll_target, wait_subject):
            if candidate and not Normalizer.is_generic_target_name(candidate):
                return Normalizer.clean(text=candidate)

        action_type = get_action_type(step=step)
        inferred = infer_target_from_rationale(
            action_type=action_type,
            rationale=rationale,
            fallback=resolved_target,
        )
        if inferred and not Normalizer.is_generic_target_name(inferred):
            return inferred

        return "element"

    if should_generalize_target(rationale=rationale):
        generalized = generalize_product_target(
            target=resolved_target,
            rationale=rationale,
        )
        if generalized != resolved_target:
            return generalized

    return resolved_target
