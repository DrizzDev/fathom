from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Union

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
        Uses the pre-computed 'generalized_target' if available (dynamic).
        Otherwise uses the specific target text (stable).
        """
        if isinstance(step, StepResult):
            if step.generalized_target:
                return step.generalized_target
            target = step.step.action.natural_language_target or step.step.action.target
        else:
            if step.get("generalized_target"):
                return str(step.get("generalized_target") or "")
            target = step.get("natural_language_target") or step.get("target") or ""

        return target or "element"

    _SWIPE_ACTIONS = {"swipe_up", "swipe_down", "swipe_left", "swipe_right"}

    @staticmethod
    def _get_action_type(step: Union[StepResult, Dict[str, Any]]) -> str:
        """Extract the action type string from a step."""
        if isinstance(step, StepResult):
            return step.step.action.action_type.value
        return step.get("action_type", "unknown")

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
        }
        return mapping.get(action_type, "Scroll")

    @staticmethod
    def export(
        step_results: Sequence[Union[StepResult, Dict[str, Any]]],
        goal_state: str = "",
    ) -> str:
        """
        Export steps to a natural language test script.

        Consecutive swipe actions are collapsed into a single
        'Scroll down until X is visible' instruction where X is the
        target of the next non-swipe action.

        Args:
            step_results: List of executed step results.
            goal_state: Optional specific goal state for final validation.
        """
        lines: list[str] = []
        n = len(step_results)
        i = 0

        while i < n:
            step = step_results[i]
            action_type_val = ScriptExporter._get_action_type(step)

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
                    and target.lower() not in ("element", "ui element", "none", "a visible item")
                ):
                    lines.append(f"Validate {target} is visible")

            # --- Build action description ---
            if isinstance(step, StepResult):
                action = step.step.action
                description = ScriptExporter._build_description(
                    action_type_val, target, action.text
                )
            else:
                text = step.get("text")
                description = ScriptExporter._build_description(action_type_val, target, text)

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
                    if last_target and last_target.lower() not in (
                        "element",
                        "ui element",
                        "none",
                        "a visible item",
                    ):
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
