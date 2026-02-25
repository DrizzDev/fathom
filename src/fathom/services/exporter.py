from __future__ import annotations

import re
from typing import Any, Dict, Optional, Sequence, Union

from fathom.schemas.steps import StepResult
from fathom.services.text_normalization import (
    clean_text,
    describe_action,
    describe_validation,
    normalize_wait_condition,
)

_ORDINAL_MAP = {
    "1st": "first",
    "2nd": "second",
    "3rd": "third",
    "4th": "fourth",
    "5th": "fifth",
    "6th": "sixth",
    "7th": "seventh",
    "8th": "eighth",
    "9th": "ninth",
    "10th": "tenth",
}

_NUMERIC_ORDINAL_RE = re.compile(r"\b(\d+)(?:st|nd|rd|th)\b", re.IGNORECASE)


_GENERIC_TARGETS = frozenset({"element", "ui element", "none", "a visible item"})


class ScriptExporter:
    """
    Service for exporting execution history to natural language scripts.
    """

    @staticmethod
    def _normalize_positional(target: str) -> str:
        """Standardize ordinal formatting in positional target descriptions.

        Converts numeric ordinals (1st, 2nd, 3rd) to word ordinals
        (first, second, third) and ensures a leading 'the' article
        for consistency. Only transforms targets that look positional.

        Examples:
            '1st search result'       -> 'the first search result'
            'the 2nd card'            -> 'the second card'
            'third item in the list'  -> 'the third item in the list'
            'Submit button'           -> 'Submit button'  (unchanged)
        """
        if not target:
            return target

        text = target.strip()

        def _replace_numeric(match: "re.Match[str]") -> str:
            full = match.group(0).lower()
            return _ORDINAL_MAP.get(full, full)

        normalized = _NUMERIC_ORDINAL_RE.sub(_replace_numeric, text)

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
        stripped = re.sub(r"^(?:the|a|an)\s+", "", normalized, flags=re.IGNORECASE).strip().lower()
        is_positional = any(stripped.startswith(o) for o in word_ordinals)

        if not is_positional:
            return target

        without_article = re.sub(r"^(?:the|a|an)\s+", "", normalized, flags=re.IGNORECASE).strip()
        return f"the {without_article}"

    @staticmethod
    def _is_intent_target(target: str, intent: str) -> bool:
        """
        Check if a target was mentioned or implied by the user's intent.
        Uses word overlap to handle fuzzy matches.
        """
        if not target or not intent:
            return False

        target_lower = target.lower()
        intent_lower = intent.lower()

        if target_lower in intent_lower:
            return True

        # Word overlap against the intent
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
            return True  # Only filler words = generic enough already

        intent_words = set(intent_lower.replace("_", " ").split())
        overlap = meaningful & intent_words
        return len(overlap) >= len(meaningful) * 0.5

    @staticmethod
    def _resolve_target(step: Union[StepResult, Dict[str, Any]]) -> str:
        """
        Resolve the description for a target.
        Uses the pre-computed 'generalized_target' if available (dynamic/positional).
        Otherwise uses the specific target text (stable).
        Normalizes positional ordinals for consistency.
        """
        if isinstance(step, StepResult):
            if step.generalized_target:
                return ScriptExporter._normalize_positional(step.generalized_target)
            target = step.step.action.natural_language_target or step.step.action.target
        else:
            if step.get("generalized_target"):
                raw = str(step.get("generalized_target") or "")
                return ScriptExporter._normalize_positional(raw)
            target = step.get("natural_language_target") or step.get("target") or ""

        return target or "element"

    _SWIPE_ACTIONS = {"swipe_up", "swipe_down", "swipe_left", "swipe_right", "scroll"}

    @staticmethod
    def _get_event_type(step: Union[StepResult, Dict[str, Any]]) -> str:
        """Extract semantic event type from a step."""
        if isinstance(step, StepResult):
            return step.step.event_type or "action"
        return str(step.get("event_type") or "action")

    @staticmethod
    def _get_action_type(step: Union[StepResult, Dict[str, Any]]) -> str:
        """Extract the action type string from a step."""
        if isinstance(step, StepResult):
            return step.step.action.action_type.value
        return str(step.get("action_type", "unknown"))

    @staticmethod
    def _swipe_direction_label(action_type: str) -> str:
        """
        Map a swipe action to its user-facing scroll/swipe label.
        swipe_up   -> 'Scroll down'  (finger moves up = content scrolls down)
        swipe_down -> 'Scroll up'
        swipe_left -> 'Swipe left'
        swipe_right -> 'Swipe right'
        """
        mapping = {
            "swipe_up": "Scroll down",
            "swipe_down": "Scroll up",
            "swipe_left": "Swipe left",
            "swipe_right": "Swipe right",
            "scroll": "Scroll down",
        }
        return mapping.get(action_type, "Scroll")

    @staticmethod
    def _get_activity(step: Union[StepResult, Dict[str, Any]]) -> str:
        """Extract the activity string from a step (empty string if absent)."""
        if isinstance(step, dict):
            return str(step.get("activity") or "")
        return ""

    @staticmethod
    def _find_app_launch_boundary(
        steps: Sequence[Union[StepResult, Dict[str, Any]]],
        package_name: str,
    ) -> int:
        """Find the index of the first step that runs inside the target package.

        Returns the index of the first step whose ``activity`` starts with
        *package_name*, or ``0`` if the app is already active from the start
        (or activity data is unavailable).  All steps before this index are
        considered "opening the app" actions and should be suppressed in
        favour of a single ``OPEN_APP`` directive.

        A safety cap of 10 prevents suppressing large portions of the history
        if something unexpected happened.
        """
        _MAX_LAUNCH_STEPS = 10
        prefix = package_name + "/"
        for j, step in enumerate(steps):
            if j > _MAX_LAUNCH_STEPS:
                return 0
            activity = ScriptExporter._get_activity(step)
            if activity.startswith(prefix) or activity == package_name:
                return j
        return 0

    @staticmethod
    def export(
        step_results: Sequence[Union[StepResult, Dict[str, Any]]],
        goal_state: str = "",
        package_name: str = "",
        intent: str = "",
    ) -> str:
        """
        Export steps to a natural language test script.

        Consecutive swipe actions are collapsed into a single
        'Scroll down until X is visible' instruction where X is the
        target of the next non-swipe action.

        Args:
            step_results: List of executed step results.
            goal_state: Optional specific goal state for final validation.
            package_name: Android package name. When set, all steps that
                precede the first in-app activity are suppressed and replaced
                with a single ``OPEN_APP <package_name>`` directive.
        """
        lines: list[str] = []
        n = len(step_results)
        i = 0
        launch_boundary = 0
        swipe_just_processed = False  # Track when we finish processing a swipe sequence
        validation_subjects = ScriptExporter._extract_validation_subjects(intent)
        validation_subject_index = 0
        emitted_validation_lines: set[str] = set()

        # Detect app-launch boundary: suppress all pre-launch steps and emit OPEN_APP
        if package_name:
            launch_boundary = ScriptExporter._find_app_launch_boundary(step_results, package_name)
            if launch_boundary > 0:
                lines.append(f"OPEN_APP {package_name}")
                i = launch_boundary

        while i < n:
            step = step_results[i]
            action_type_val = ScriptExporter._get_action_type(step)
            event_type = ScriptExporter._get_event_type(step)
            condition = ScriptExporter._get_condition(step)

            # --- Detect start of a swipe sequence ---
            if action_type_val in ScriptExporter._SWIPE_ACTIONS:
                swipe_direction = action_type_val
                # Skip all consecutive swipes of the SAME direction only
                j = i + 1
                while j < n and ScriptExporter._get_action_type(step_results[j]) == swipe_direction:
                    j += 1
                i = j

                # Find the target of the NEXT non-swipe step (lookahead)
                if i < n:
                    next_target = ScriptExporter._resolve_target(step_results[i])
                else:
                    next_target = goal_state or "the target"

                label = ScriptExporter._swipe_direction_label(swipe_direction)
                lines.append(f"{label} until {next_target} is visible")
                swipe_just_processed = True  # Mark that we just processed swipes
                continue  # don't increment i again, already advanced

            # --- Resolve target for current step ---
            target = ScriptExporter._resolve_target(step)

            # --- Smart Validation: insert when previous step caused a screen change ---
            # Skip validation immediately after swipe sequences (swipe statement already includes "until X is visible")
            if (
                i > 0
                and i > launch_boundary
                and action_type_val != "wait"
                and not swipe_just_processed
                and event_type != "validation"
                and not condition
            ):
                prev = step_results[i - 1]
                prev_action_type = ScriptExporter._get_action_type(prev)
                prev_condition = ScriptExporter._get_condition(prev)
                prev_changed = (
                    prev.screen_changed
                    if isinstance(prev, StepResult)
                    else prev.get("screen_changed", False)
                )
                if (
                    prev_changed
                    and prev_action_type not in ("wait", *ScriptExporter._SWIPE_ACTIONS)
                    and not prev_condition
                    and target.lower() not in _GENERIC_TARGETS
                ):
                    val_line = f"Validate {target} is visible"
                    lines.append(val_line)

            swipe_just_processed = False  # Reset flag for non-swipe actions

            if isinstance(step, StepResult):
                action = step.step.action
                text = action.text
                rationale = action.rationale
            else:
                text = step.get("text")
                rationale = str(step.get("rationale", ""))

            # For wait actions with generic targets, derive a better subject
            if action_type_val == "wait" and target.lower() in _GENERIC_TARGETS:
                target = ScriptExporter._infer_wait_subject(rationale)

            description = ScriptExporter._build_description(action_type_val, target, text)

            if event_type == "validation":
                effective_target = target
                is_system_validation = ScriptExporter._is_system_validation(
                    target=target, rationale=rationale, condition=condition
                )
                should_use_intent_subject = (
                    bool(validation_subjects)
                    and not is_system_validation
                    and validation_subject_index < len(validation_subjects)
                )
                if should_use_intent_subject:
                    effective_target = validation_subjects[
                        min(validation_subject_index, len(validation_subjects) - 1)
                    ]
                    validation_subject_index += 1
                validation_condition = ScriptExporter._infer_validation_condition(
                    condition=condition,
                    action_type=action_type_val,
                    target=effective_target,
                    rationale=rationale,
                )
                if not validation_condition and i > 0 and not should_use_intent_subject:
                    prev = step_results[i - 1]
                    prev_condition = ScriptExporter._get_condition(prev)
                    prev_action_type = ScriptExporter._get_action_type(prev)
                    if prev_condition and prev_action_type == "wait":
                        validation_condition = prev_condition
                validation_line = ScriptExporter._build_validation_description(
                    action_type=action_type_val,
                    target=effective_target,
                    use_explicit_phrase=should_use_intent_subject,
                )
                # If the very next step is already an IF-wrapped action on the same
                # target/condition, skip this standalone conditional validation line.
                if validation_condition and i + 1 < n:
                    next_step = step_results[i + 1]
                    next_condition = ScriptExporter._get_condition(next_step)
                    next_target = ScriptExporter._resolve_target(next_step)
                    next_event_type = ScriptExporter._get_event_type(next_step)
                    if (
                        next_event_type != "validation"
                        and next_condition == validation_condition
                        and next_target.strip().lower() == target.strip().lower()
                    ):
                        i += 1
                        continue
                if validation_line in emitted_validation_lines:
                    i += 1
                    continue
                if validation_condition:
                    # If previous wait already emitted the same IF block,
                    # append this validation into that block instead of
                    # creating another repeated single-line IF.
                    merged_into_previous_wait_block = False
                    if i > 0:
                        prev = step_results[i - 1]
                        prev_action_type = ScriptExporter._get_action_type(prev)
                        prev_condition = ScriptExporter._get_condition(prev)
                        if (
                            prev_action_type == "wait"
                            and prev_condition == validation_condition
                            and len(lines) >= 3
                            and lines[-3] == f"IF {validation_condition} {{"
                            and lines[-1] == "}"
                        ):
                            lines.pop()  # remove closing brace
                            lines.append(f"    {validation_line}")
                            lines.append("}")
                            merged_into_previous_wait_block = True
                    if merged_into_previous_wait_block:
                        emitted_validation_lines.add(validation_line)
                        i += 1
                        continue
                    conditional_line = f"IF {validation_condition} {{ {validation_line} }}"
                    lines.append(conditional_line)
                else:
                    lines.append(validation_line)
                emitted_validation_lines.add(validation_line)
                i += 1
                continue

            # Wrap in IF block with Pre-Action Validation
            if condition:
                lines.append(f"IF {condition} {{")
                prev_is_same_target_validation = False
                if i > 0:
                    prev = step_results[i - 1]
                    prev_event_type = ScriptExporter._get_event_type(prev)
                    prev_target = ScriptExporter._resolve_target(prev)
                    prev_is_same_target_validation = (
                        prev_event_type == "validation"
                        and prev_target.strip().lower() == target.strip().lower()
                    )
                if (
                    target.lower() not in _GENERIC_TARGETS
                    and action_type_val != "wait"
                    and not prev_is_same_target_validation
                ):
                    lines.append(f"    Validate {target} is visible")
                lines.append(f"    {description}")
                lines.append("}")
            else:
                lines.append(description)
            i += 1

        # --- Ensure final step is a validation ---
        if step_results:
            last_action_type = ScriptExporter._get_action_type(step_results[-1])

            if last_action_type not in ("complete", "verify_goal_completion"):
                if validation_subjects:
                    final_validation_line = f"Validate that {validation_subjects[-1]}"
                    if final_validation_line not in emitted_validation_lines and (
                        not lines or lines[-1].strip() != final_validation_line
                    ):
                        lines.append(final_validation_line)
                        emitted_validation_lines.add(final_validation_line)
                    return "\n".join(lines) + "\n"
                if goal_state:
                    final_goal_line = f"Validate {goal_state}"
                    if not lines or lines[-1].strip() != final_goal_line:
                        lines.append(final_goal_line)
                else:
                    last_target = ScriptExporter._resolve_target(step_results[-1])
                    if last_target and last_target.lower() not in _GENERIC_TARGETS:
                        final_target_line = f"Validate {last_target}"
                        if not lines or lines[-1].strip() != final_target_line:
                            lines.append(final_target_line)
                    else:
                        final_default_line = "Validate Goal State"
                        if not lines or lines[-1].strip() != final_default_line:
                            lines.append(final_default_line)

        return "\n".join(lines) + "\n"

    @staticmethod
    def _build_description(action_type: str, target: str, text: Optional[str] = None) -> str:
        """Build a human-readable action description."""
        return describe_action(action_type=action_type, target=target, text=text)

    @staticmethod
    def _build_validation_description(
        action_type: str, target: str, use_explicit_phrase: bool = False
    ) -> str:
        """Build an explicit validation-only description."""
        return describe_validation(
            target=target,
            explicit=use_explicit_phrase,
            complete=(action_type == "complete"),
        )

    @staticmethod
    def _is_system_validation(
        *, target: str, rationale: Optional[str], condition: Optional[str]
    ) -> bool:
        """Detect transient/blocker validations that should not consume intent subjects."""
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

    @staticmethod
    def _normalize_wait_condition_phrase(
        condition: Optional[str], rationale: Optional[str]
    ) -> Optional[str]:
        """Normalize awkward wait-condition phrases into clearer wording."""
        return normalize_wait_condition(condition=condition, rationale=rationale)

    @staticmethod
    def _extract_validation_subjects(intent: str) -> list[str]:
        """Extract all user-requested validation subjects from intent text, in order."""
        if not intent:
            return []
        matches = re.finditer(
            r"\b(?:validate|verify|check|confirm)(?:\s+that)?\s+(.+?)(?=(?:,|\bthen\b|\band\s+(?:validate|verify|check|confirm)\b|$))",
            intent,
            flags=re.IGNORECASE,
        )
        subjects: list[str] = []
        for match in matches:
            subject = clean_text(match.group(1).strip(" .,:;"))
            if subject:
                subjects.append(subject)
        return subjects

    @staticmethod
    def _is_precondition_validation_subject(subject: str) -> bool:
        """Heuristic for validation subjects that should be checked early."""
        if not subject:
            return False
        lower = subject.lower()
        return any(
            token in lower
            for token in ("logged in", "logged-in", "signed in", "authenticated", "account")
        )

    @staticmethod
    def _infer_wait_subject(rationale: Optional[str]) -> str:
        """Derive a human-readable wait subject from the step rationale.

        Used when the VLM produced a generic target like "UI Element" for
        a wait action.  Returns a concise phrase suitable for both the IF
        condition and the Wait description.
        """
        if not rationale:
            return "screen to load"

        lower = str(rationale).lower()

        if re.search(r"\bad\b", lower) and (
            "play" in lower or "finish" in lower or "skip" in lower
        ):
            return "ad to finish"

        if "splash" in lower or "load" in lower or "main interface" in lower:
            return "app to finish loading"

        return "screen to load"

    @staticmethod
    def _infer_validation_condition(
        *,
        condition: Optional[str],
        action_type: str,
        target: str,
        rationale: Optional[str],
    ) -> Optional[str]:
        """Infer IF conditions for validation events on transient/blocker screens."""
        lower = str(rationale or "").lower()
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
            if target.lower() in _GENERIC_TARGETS:
                return f"{ScriptExporter._infer_wait_subject(rationale)} is visible"
            return f"{target} is visible"
        return None

    @staticmethod
    def _get_condition(step: Union[StepResult, Dict[str, Any]]) -> Optional[str]:
        """
        Get the condition for a step, inferring from rationale if needed.
        Uses resolved (generalized) target for wait conditions to avoid
        generic fallbacks like "UI Element".
        """
        condition: Optional[str] = None
        rationale: Optional[str] = None
        action_type: str = "wait"

        if isinstance(step, StepResult):
            condition = getattr(step.step, "condition", None) or getattr(
                step.step.action, "condition", None
            )
            rationale = clean_text(step.step.action.rationale)
            action_type = step.step.action.action_type.value.lower()
        else:
            condition = clean_text(step.get("condition"))
            rationale = clean_text(step.get("rationale"))
            action_type = str(step.get("action_type", "wait")).lower()

        # Heuristic Inference
        if not condition and rationale:
            lower_rationale = str(rationale).lower()
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
            if "timeout" in lower_rationale:
                condition = "Timeout error is displayed"
            elif (
                "retry" in lower_rationale
                or "try again" in lower_rationale
                or "error" in lower_rationale
            ):
                condition = "Error message is displayed"

        # Enforce Conditional Wait
        if action_type == "wait" and not condition:
            resolved = ScriptExporter._resolve_target(step)
            if resolved.lower() in _GENERIC_TARGETS:
                subject = ScriptExporter._infer_wait_subject(rationale)
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
            condition = ScriptExporter._normalize_wait_condition_phrase(condition, rationale)

        return condition
