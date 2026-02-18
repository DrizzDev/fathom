from __future__ import annotations

import re
from typing import Any, Dict, Optional, Sequence, Union

from fathom.schemas.steps import StepResult

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
    def _is_app_launch(step: Union[StepResult, Dict[str, Any]], index: int) -> bool:
        """Detect whether a step is the initial app-launch tap."""
        if index != 0:
            return False
        action_type = ScriptExporter._get_action_type(step)
        if action_type != "tap":
            return False
        if isinstance(step, StepResult):
            raw = (
                step.step.action.natural_language_target or step.step.action.target or ""
            ).lower()
        else:
            raw = (step.get("natural_language_target") or step.get("target") or "").lower()
        return "app" in raw or "icon" in raw

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
            package_name: Android package name. When set, the first tap-on-
                app-icon step is replaced with ``OPEN_APP <package_name>``.
        """
        lines: list[str] = []
        n = len(step_results)
        i = 0

        while i < n:
            step = step_results[i]
            action_type_val = ScriptExporter._get_action_type(step)

            # Replace the app-launch tap with a deterministic OPEN_APP command
            if package_name and ScriptExporter._is_app_launch(step, i):
                lines.append(f"OPEN_APP {package_name}")
                i += 1
                continue

            # --- Detect start of a swipe sequence ---
            if action_type_val in ScriptExporter._SWIPE_ACTIONS:
                swipe_direction = action_type_val
                # Skip all consecutive swipes of the same direction
                while (
                    i < n
                    and ScriptExporter._get_action_type(step_results[i])
                    in ScriptExporter._SWIPE_ACTIONS
                ):
                    i += 1

                # Find the target of the NEXT non-swipe step (lookahead)
                if i < n:
                    next_target = ScriptExporter._resolve_target(step_results[i])
                else:
                    next_target = goal_state or "the target"

                label = ScriptExporter._swipe_direction_label(swipe_direction)
                lines.append(f"{label} until {next_target} is visible")
                continue  # don't increment i again, already advanced

            # --- Resolve target for current step ---
            target = ScriptExporter._resolve_target(step)

            # --- Smart Validation: insert when previous step caused a screen change ---
            if i > 0 and action_type_val != "wait":
                prev = step_results[i - 1]
                prev_action_type = ScriptExporter._get_action_type(prev)
                prev_changed = (
                    prev.screen_changed
                    if isinstance(prev, StepResult)
                    else prev.get("screen_changed", False)
                )
                if (
                    prev_changed
                    and prev_action_type not in ("wait", *ScriptExporter._SWIPE_ACTIONS)
                    and target.lower() not in _GENERIC_TARGETS
                ):
                    val_line = f"Validate {target} is visible"
                    prev_condition = ScriptExporter._get_condition(prev)
                    if prev_condition:
                        val_line = f"IF {prev_condition} {{ {val_line} }}"
                    lines.append(val_line)

            condition = ScriptExporter._get_condition(step)

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

            # Wrap in IF block with Pre-Action Validation
            if condition:
                lines.append(f"IF {condition} {{")
                if target.lower() not in _GENERIC_TARGETS and action_type_val != "wait":
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
                if goal_state:
                    lines.append(f"Validate {goal_state} is visible")
                else:
                    last_target = ScriptExporter._resolve_target(step_results[-1])
                    if last_target and last_target.lower() not in _GENERIC_TARGETS:
                        lines.append(f"Validate {last_target} is visible")
                    else:
                        lines.append("Validate Goal State is visible")

        return "\n".join(lines) + "\n"

    @staticmethod
    def _build_description(action_type: str, target: str, text: Optional[str] = None) -> str:
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

        if "ad" in lower and ("play" in lower or "finish" in lower or "skip" in lower):
            return "ad to finish"

        if "splash" in lower or "load" in lower or "main interface" in lower:
            return "app to finish loading"

        return "screen to load"

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
            rationale = step.step.action.rationale
            action_type = step.step.action.action_type.value.lower()
        else:
            condition = step.get("condition")
            rationale = step.get("rationale")
            action_type = str(step.get("action_type", "wait")).lower()

        # Heuristic Inference
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

        # Enforce Conditional Wait
        if action_type == "wait" and not condition:
            resolved = ScriptExporter._resolve_target(step)
            if resolved.lower() in _GENERIC_TARGETS:
                subject = ScriptExporter._infer_wait_subject(rationale)
                condition = f"{subject} is visible"
            else:
                condition = f"{resolved} is visible"

        return condition
