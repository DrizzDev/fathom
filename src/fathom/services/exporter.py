from __future__ import annotations

from typing import Any, Dict, List, Union

from fathom.schemas.steps import StepResult


class ScriptExporter:
    """
    Service for exporting execution history to natural language scripts.
    """

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
        filler = {"the", "a", "an", "on", "in", "to", "of", "is", "and", "or",
                  "item", "button", "icon", "area", "field"}
        meaningful = target_words - filler
        if not meaningful:
            return True  # Only filler words = generic enough already

        intent_words = set(intent_lower.replace("_", " ").split())
        overlap = meaningful & intent_words
        return len(overlap) >= len(meaningful) * 0.5

    @staticmethod
    def export(
        step_results: List[Union[StepResult, Dict[str, Any]]],
        intent: str = "",
    ) -> str:
        """
        Export steps to a natural language test script.
        Uses the structured Action data directly for clean, readable output.
        Includes Smart Validation for screen changes.

        Targets not mentioned in the user's original intent are generalized
        to ensure the script is reproducible across different runs.

        Args:
            step_results: List of executed step results.
            intent: The original user intent, used to determine which targets
                    are stable (mentioned by user) vs dynamic (content-specific).
        """
        lines = []
        step_num = 1

        for i, step in enumerate(step_results):
            # --- Extract raw target and action info ---
            if isinstance(step, StepResult):
                action = step.step.action
                raw_target = action.natural_language_target or action.target
                action_type_val = action.action_type.value
                screen_changed = step.screen_changed
            else:
                raw_target = step.get("natural_language_target") or step.get("target") or "element"
                action_type_val = step.get("action_type", "unknown")
                screen_changed = step.get("screen_changed", False)

            # --- Resolve target: keep intent-mentioned targets, generalize the rest ---
            if ScriptExporter._is_intent_target(raw_target, intent):
                target = raw_target
            else:
                target = "a visible item" if action_type_val in ("tap", "long_press") else "the current view"

            # --- Smart Validation: insert when previous step caused a screen change ---
            if i > 0:
                prev = step_results[i - 1]
                prev_changed = (
                    prev.screen_changed if isinstance(prev, StepResult) else prev.get("screen_changed", False)
                )
                if prev_changed:
                    if target.lower() not in ("element", "ui element", "none", "a visible item"):
                        lines.append(f"{step_num}. Validate {target} is visible")
                        step_num += 1

            # --- Build action description ---
            if isinstance(step, StepResult):
                action = step.step.action
                description = ScriptExporter._build_description(action_type_val, target, action.text)
            else:
                text = step.get("text")
                description = ScriptExporter._build_description(action_type_val, target, text)

            lines.append(f"{step_num}. {description}")
            step_num += 1

        return "\n".join(lines) + "\n"

    @staticmethod
    def _build_description(action_type: str, target: str, text: str = None) -> str:
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
