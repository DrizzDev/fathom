from __future__ import annotations

import re
from typing import Any, Dict, Optional, Sequence, Union

from fathom.core.services.exporter.constants import (
    GENERIC_TARGETS,
    PROPER_PHRASE_RE,
    SCROLL_VERB_RE,
)
from fathom.core.services.exporter.step_targets import resolve_target
from fathom.core.services.normalizer import Normalizer
from fathom.schemas.steps import StepResult


def is_system_validation(
    *, target: str, rationale: Optional[str], condition: Optional[str]
) -> bool:
    signal = " ".join([target or "", rationale or "", condition or ""]).lower()
    system_terms = (
        "overlay",
        "popup",
        "pop-up",
        "dialog",
        "permission",
        "consent",
        "cookie",
        "splash",
        "loading",
        "spinner",
        "interstitial",
        "close button",
        "got it",
        "blocker",
        "transient",
    )
    return any(term in signal for term in system_terms)


def infer_scroll_target(
    steps: Sequence[Union[StepResult, Dict[str, Any]]],
    start: int,
    end: int,
) -> str:
    for j in range(start, min(end, start + 5)):
        step = steps[j]
        if isinstance(step, StepResult):
            scroll_target = step.step.action.scroll_target
            if scroll_target:
                return scroll_target
        else:
            scroll_target = step.get("scroll_target")
            if scroll_target:
                return str(scroll_target)

    for j in range(start, min(end, start + 5)):
        step = steps[j]
        if isinstance(step, StepResult):
            rationale = step.step.action.rationale or ""
        else:
            rationale = str(object=step.get("rationale") or "")
        if not rationale:
            continue
        verb_match = SCROLL_VERB_RE.search(string=rationale)
        if not verb_match:
            continue
        clause = verb_match.group(1).strip()

        quoted_match = re.search(r"['\"]([^'\"]+)['\"]", clause)
        if quoted_match:
            extracted = quoted_match.group(1).strip()
            extracted = re.sub(
                r"\s+(section|category|area|page|screen|button|tab|widget)$",
                "",
                extracted,
                flags=re.IGNORECASE,
            )
            if extracted:
                return extracted

        product_match = PROPER_PHRASE_RE.search(string=clause)
        if product_match:
            return product_match.group(1).strip()

        cleaned = re.sub(r"^the\s+", "", clause, flags=re.IGNORECASE)
        cleaned = re.sub(
            r"\s+(section|category|area|page|screen|button|tab|widget)$",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = cleaned.strip(" ,'\"")
        if cleaned and len(cleaned) > 5:
            return cleaned
    return ""


def infer_wait_subject(rationale: Optional[str], wait_subject: Optional[str] = None) -> str:
    if wait_subject:
        return wait_subject

    return Normalizer.wait_subject(rationale=rationale) or "screen to load"


def infer_validation_condition(
    *,
    target: str,
    action_type: str,
    rationale: Optional[str],
    condition: Optional[str],
) -> Optional[str]:
    lower = str(object=rationale or "").lower()
    blocker_terms = ("permission", "cookie", "consent", "popup", "dialog", "blocker")
    transient_terms = (
        "loading",
        "spinner",
        "splash",
        "interstitial",
        "ad",
        "please wait",
    )

    if any(term in lower for term in blocker_terms):
        return "Blocker prompt is visible"

    if any(term in lower for term in transient_terms):
        return "Transient screen is visible"

    if condition:
        return condition

    if action_type == "wait":
        if target.lower() in GENERIC_TARGETS:
            return f"{infer_wait_subject(rationale=rationale)} is visible"

        return f"{target} is visible"

    return None


def is_blocker_popup_condition(condition: Optional[str]) -> bool:
    if not condition:
        return False

    signal = condition.lower()
    blocker_terms = (
        "blocker",
        "popup",
        "pop-up",
        "overlay",
        "prompt",
        "dialog",
        "notification",
        "permission",
        "consent",
        "cookie",
        "transient",
        "interstitial",
    )
    return any(term in signal for term in blocker_terms)


def get_condition(step: Union[StepResult, Dict[str, Any]]) -> Optional[str]:
    condition: Optional[str] = None
    rationale: Optional[str] = None
    action_type: str = "wait"

    if isinstance(step, StepResult):
        condition = getattr(step.step, "condition", None) or getattr(
            step.step.action, "condition", None
        )
        rationale = Normalizer.clean(text=step.step.action.rationale)
        action_type = step.step.action.action_type.value.lower()
    else:
        condition = Normalizer.clean(text=step.get("condition"))
        rationale = Normalizer.clean(text=step.get("rationale"))
        action_type = str(object=step.get("action_type", "wait")).lower()

    if not condition and rationale:
        lower_rationale = str(object=rationale).lower()
        if (
            "overlay" in lower_rationale
            or "popup" in lower_rationale
            or "pop-up" in lower_rationale
        ) and (
            "dismiss" in lower_rationale
            or "close" in lower_rationale
            or "skip" in lower_rationale
            or "got it" in lower_rationale
        ):
            condition = "Promotional overlay is visible"
        elif any(
            token in lower_rationale
            for token in ("prompt", "permission", "dialog", "consent", "cookie")
        ) and any(
            token in lower_rationale
            for token in (
                "dismiss",
                "close",
                "skip",
                "not now",
                "deny",
                "allow",
                "accept",
                "continue",
            )
        ):
            condition = "Blocker prompt is visible"
        if "timeout" in lower_rationale:
            condition = "Timeout error is displayed"
        elif (
            "retry" in lower_rationale
            or "try again" in lower_rationale
            or "error" in lower_rationale
        ):
            condition = "Error message is displayed"

    if action_type == "wait" and not condition:
        resolved = resolve_target(step=step)
        if resolved.lower() in GENERIC_TARGETS:
            subject = infer_wait_subject(rationale=rationale)
            if subject == "app to finish loading":
                condition = "the app is still loading"
            else:
                condition = f"{subject} is visible"
        else:
            resolved_lower = resolved.lower()
            if "search result" in resolved_lower or "results" in resolved_lower:
                condition = "search results are still loading"
            else:
                condition = f"{resolved} is visible"

    if action_type == "wait":
        condition = Normalizer.wait_condition(condition=condition, rationale=rationale)

    return condition
