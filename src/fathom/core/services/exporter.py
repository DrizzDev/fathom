from __future__ import annotations

import re
from typing import Any, Dict, Optional, Sequence, Union

from fathom.schemas.steps import StepResult

ORDINAL_MAP = {
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

NUMERIC_ORDINAL_RE = re.compile(r"\b(\d+)(?:st|nd|rd|th)\b", re.IGNORECASE)
GENERIC_TARGETS = frozenset({"element", "ui element", "none", "a visible item"})


class ScriptExporter:
    """
    Service for exporting execution history to natural language scripts.
    """

    @staticmethod
    def __normalize_positional(target: str) -> str:
        """
        Standardize ordinal formatting in positional target descriptions.

        Converts numeric ordinals (1st, 2nd, 3rd) to word ordinals
        (first, second, third) and ensures a leading 'the' article
        for consistency. Only transforms targets that look positional.

        Examples:
            'The 2nd card'            -> 'the second card'
            '1st search result'       -> 'the first search result'
            'Third item in the list'  -> 'the third item in the list'
            'Submit button'           -> 'Submit button'  (unchanged)
        """

        if not target:
            return target

        text = target.strip()

        def __replace_numeric(match: "re.Match[str]") -> str:
            full = match.group(0).lower()
            return ORDINAL_MAP.get(full, full)

        normalized = NUMERIC_ORDINAL_RE.sub(__replace_numeric, text)

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
    def __is_intent_target(target: str, intent: str) -> bool:
        """
        Check if a target was mentioned or implied by the user's intent. Uses word overlap to handle fuzzy matches.
        """

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

    __SCREEN_RE = re.compile(
        r"(?:the\s+)?(\w+(?:\s+\w+)?)\s+(screen|page)\b",
        re.IGNORECASE,
    )
    __LABEL_STOP = frozenset(
        {
            "a",
            "an",
            "the",
            "any",
            "some",
            "no",
            "or",
            "and",
            "this",
            "that",
            "on",
            "in",
            "at",
            "to",
            "of",
            "is",
            "it",
            "my",
            "its",
        }
    )

    @staticmethod
    def __extract_goal_label(goal_state: str) -> str:
        """
        Derive a concise validation label from a potentially long intent.

        Short strings (<=60 chars, no sentence-ending punctuation) are returned as-is.
        Longer intent strings are scanned for a "screen/page" reference which is title-cased into a label like "Payment Screen".
        Returns empty string when nothing useful can be extracted.
        """

        if not goal_state:
            return ""

        trimmed = goal_state.strip().rstrip(".")
        if len(trimmed) <= 60 and "." not in trimmed:
            return trimmed

        matches = ScriptExporter.__SCREEN_RE.findall(goal_state)

        for name, kind in reversed(matches):
            cleaned = name.strip()
            words = cleaned.lower().split()

            if len(cleaned) > 1 and not any(w in ScriptExporter.__LABEL_STOP for w in words):
                return f"{cleaned.title()} {kind.strip().title()}"

        return ""

    @staticmethod
    def __resolve_target(step: Union[StepResult, Dict[str, Any]]) -> str:
        """
        Resolve the description for a target.
        Uses the pre-computed 'generalized_target' if available (dynamic/positional).
        Otherwise uses the specific target text (stable). Normalizes positional ordinals for consistency.
        """

        if isinstance(step, StepResult):
            if step.generalized_target:
                return ScriptExporter.__normalize_positional(step.generalized_target)

            target = step.step.action.natural_language_target or step.step.action.target

        else:
            if step.get("generalized_target"):
                raw = str(step.get("generalized_target") or "")
                return ScriptExporter.__normalize_positional(raw)

            target = step.get("natural_language_target") or step.get("target") or ""

        return target or "element"

    __SWIPE_ACTIONS = {"swipe_up", "swipe_down", "swipe_left", "swipe_right", "scroll"}

    @staticmethod
    def __get_action_type(step: Union[StepResult, Dict[str, Any]]) -> str:
        """
        Extract the action type string from a step.
        """

        if isinstance(step, StepResult):
            return step.step.action.action_type.value

        return str(step.get("action_type", "unknown"))

    @staticmethod
    def __swipe_direction_label(action_type: str) -> str:
        """
        Map a swipe action to its user-facing scroll/swipe label.

        swipe_up   -> 'Scroll down'  (finger moves up = content scrolls down)
        swipe_down -> 'Scroll up'
        swipe_left -> 'Swipe left'
        swipe_right -> 'Swipe right'
        """

        mapping = {
            "scroll": "Scroll down",
            "swipe_up": "Scroll down",
            "swipe_down": "Scroll up",
            "swipe_left": "Swipe left",
            "swipe_right": "Swipe right",
        }
        return mapping.get(action_type, "Scroll")

    @staticmethod
    def __get_activity(step: Union[StepResult, Dict[str, Any]]) -> str:
        """
        Extract the activity string from a step (empty string if absent).
        """

        if isinstance(step, dict):
            return str(step.get("activity") or "")

        return ""

    @staticmethod
    def __find_app_launch_boundary(
        steps: Sequence[Union[StepResult, Dict[str, Any]]],
        package_name: str,
    ) -> int:
        """
        Find the index of the first step that runs inside the target package.

        Returns the index of the first step whose ``activity`` starts with
        *package_name*, or ``0`` if the app is already active from the start
        (or activity data is unavailable).  All steps before this index are
        considered "opening the app" actions and should be suppressed in
        favour of a single ``OPEN_APP`` directive.

        A safety cap of 10 prevents suppressing large portions of the history
        if something unexpected happened.
        """

        __MAX_LAUNCH_STEPS = 10
        prefix = package_name + "/"

        for j, step in enumerate(steps):
            if j > __MAX_LAUNCH_STEPS:
                return 0

            activity = ScriptExporter.__get_activity(step)
            if activity.startswith(prefix) or activity == package_name:
                return j

        return 0

    @staticmethod
    def export(
        step_results: Sequence[Union[StepResult, Dict[str, Any]]],
        goal_state: str = "",
        package_name: str = "",
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
        
        filtered_results = []
        for step in step_results:
            if ScriptExporter.__get_condition(step) == "recovery":
                continue
            filtered_results.append(step)
        step_results = filtered_results
        
        n = len(step_results)
        i = 0
        launch_boundary = 0

        if package_name:
            launch_boundary = ScriptExporter.__find_app_launch_boundary(step_results, package_name)
            if launch_boundary > 0:
                lines.append(f"OPEN_APP {package_name}")
                i = launch_boundary

        while i < n:
            step = step_results[i]
            action_type_val = ScriptExporter.__get_action_type(step)

            if action_type_val in ScriptExporter.__SWIPE_ACTIONS:
                swipe_direction = action_type_val
                swipe_start = i
                while (
                    i < n
                    and ScriptExporter.__get_action_type(step_results[i])
                    in ScriptExporter.__SWIPE_ACTIONS
                ):
                    i += 1

                if i < n:
                    next_target = ScriptExporter.__resolve_target(step_results[i])
                else:
                    next_target = (
                        ScriptExporter.__infer_scroll_target(step_results, swipe_start, i)
                        or ScriptExporter.__extract_goal_label(goal_state)
                        or "the target"
                    )

                label = ScriptExporter.__swipe_direction_label(swipe_direction)
                lines.append(f"{label} until {next_target} is visible")
                continue

            target = ScriptExporter.__resolve_target(step)

            if i > 0 and i > launch_boundary and action_type_val != "wait":
                prev = step_results[i - 1]
                prev_action_type = ScriptExporter.__get_action_type(prev)
                prev_changed = (
                    prev.screen_changed
                    if isinstance(prev, StepResult)
                    else prev.get("screen_changed", False)
                )
                if (
                    prev_changed
                    and prev_action_type not in ("wait", *ScriptExporter.__SWIPE_ACTIONS)
                    and target.lower() not in GENERIC_TARGETS
                ):
                    val_line = f"Validate {target} is visible"
                    prev_condition = ScriptExporter.__get_condition(prev)
                    if prev_condition:
                        val_line = f"IF {prev_condition} {{ {val_line} }}"
                    lines.append(val_line)

            condition = ScriptExporter.__get_condition(step)

            if isinstance(step, StepResult):
                action = step.step.action
                text = action.text
                rationale = action.rationale
            else:
                text = step.get("text")
                rationale = str(step.get("rationale", ""))

            if action_type_val == "wait" and target.lower() in GENERIC_TARGETS:
                target = ScriptExporter.__infer_wait_subject(rationale)

            description = ScriptExporter.__build_description(action_type_val, target, text)

            if condition:
                lines.append(f"IF {condition} {{")
                if target.lower() not in GENERIC_TARGETS and action_type_val != "wait":
                    lines.append(f"    Validate {target} is visible")
                lines.append(f"    {description}")
                lines.append("}")
            else:
                lines.append(description)
            i += 1

        if step_results:
            last_action_type = ScriptExporter.__get_action_type(step_results[-1])

            if last_action_type not in ("complete", "verify_goal_completion"):
                last_target = ScriptExporter.__resolve_target(step_results[-1])
                last_target_usable = last_target and last_target.lower() not in GENERIC_TARGETS

                goal_label = ScriptExporter.__extract_goal_label(goal_state)

                if last_target_usable and ScriptExporter.__is_intent_target(
                    last_target, goal_state
                ):
                    lines.append(f"Validate {last_target} is visible")
                elif goal_label:
                    lines.append(f"Validate {goal_label} is visible")
                elif last_target_usable:
                    lines.append(f"Validate {last_target} is visible")
                else:
                    lines.append("Validate Goal State is visible")

        return "\n".join(lines) + "\n"

    @staticmethod
    def __build_description(action_type: str, target: str, text: Optional[str] = None) -> str:
        """Build a human-readable action description."""
        if action_type == "tap":
            return f"Tap on {target}"
        elif action_type == "type":
            return f"Type '{text}' into {target}"
        elif "swipe" in action_type:
            direction = action_type.split("_")[-1] if "_" in action_type else "content"
            return f"Swipe {direction} on {target}"
        elif action_type in ("back", "press_back"):
            return "Press back button"
        elif action_type in ("home", "press_home"):
            return "Press home button"
        elif action_type == "enter":
            return "Press enter"
        elif action_type == "wait":
            return f"Wait for {target}"
        elif action_type == "scroll":
            return f"Scroll until you see {target}"
        elif action_type == "long_press":
            return f"Long press on {target}"
        elif action_type == "complete":
            return f"Validate {target} (Goal complete)"
        else:
            return f"{action_type.replace('_', ' ').capitalize()} on {target}"

    __SCROLL_VERB_RE = re.compile(
        r"(?:find|look(?:ing)?\s+for|search(?:ing)?\s+for)\s+(.+?)(?:\.|,\s|;|$)",
        re.IGNORECASE,
    )
    __PROPER_PHRASE_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[a-z]+)*(?:\s+[A-Z][a-z]+)+)")

    @staticmethod
    def __infer_scroll_target(
        steps: Sequence[Union[StepResult, Dict[str, Any]]],
        start: int,
        end: int,
    ) -> str:
        """Infer what the user was scrolling to find from swipe rationales.

        Uses a two-pass extraction: first isolates the clause after
        "find/look for/search for", then extracts a proper-noun phrase
        (e.g. "Mango flavored Mogu Mogu") from it.
        """
        for j in range(start, min(end, start + 5)):
            step = steps[j]
            if isinstance(step, StepResult):
                rationale = step.step.action.rationale or ""
            else:
                rationale = str(step.get("rationale") or "")
            if not rationale:
                continue
            verb_match = ScriptExporter.__SCROLL_VERB_RE.search(rationale)
            if not verb_match:
                continue
            clause = verb_match.group(1)
            product_match = ScriptExporter.__PROPER_PHRASE_RE.search(clause)
            if product_match:
                return product_match.group(1).strip()
        return ""

    @staticmethod
    def __infer_wait_subject(rationale: Optional[str]) -> str:
        """Derive a human-readable wait subject from the step rationale.

        Used when the VLM produced a generic target like "UI Element" for
        a wait action.  Returns a concise phrase suitable for both the IF
        condition and the Wait description.
        """
        if not rationale:
            return "screen to load"

        lower = str(rationale).lower()

        if "ad" in lower and ("play" in lower or "finish" in lower or "skip" in lower):
            return "ad to finish"

        if "splash" in lower or "load" in lower or "main interface" in lower:
            return "app to finish loading"

        return "screen to load"

    @staticmethod
    def __get_condition(step: Union[StepResult, Dict[str, Any]]) -> Optional[str]:
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
            rationale = step.step.action.rationale
            action_type = step.step.action.action_type.value.lower()
        else:
            condition = step.get("condition")
            rationale = step.get("rationale")
            action_type = str(step.get("action_type", "wait")).lower()

        if not condition and rationale:
            lower_rationale = str(rationale).lower()
            if "timeout" in lower_rationale:
                condition = "Timeout error is displayed"
            elif (
                "retry" in lower_rationale
                or "try again" in lower_rationale
                or "error" in lower_rationale
            ):
                condition = "Error message is displayed"

        if action_type == "wait" and not condition:
            resolved = ScriptExporter.__resolve_target(step)
            if resolved.lower() in GENERIC_TARGETS:
                subject = ScriptExporter.__infer_wait_subject(rationale)
                condition = f"{subject} is visible"
            else:
                condition = f"{resolved} is visible"

        return condition
