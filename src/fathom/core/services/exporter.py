from __future__ import annotations

import re
from typing import Any, Dict, Optional, Sequence, Union

from fathom.core.services.normalizer import Normalizer
from fathom.schemas.steps import StepResult


class ScriptExporter:
    """
    Service for exporting execution history to natural language scripts.
    """

    __ORDINAL_MAP = {
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

    __NUMERIC_ORDINAL_RE = re.compile(pattern=r"\b(\d+)(?:st|nd|rd|th)\b", flags=re.IGNORECASE)
    __GENERIC_TARGETS = frozenset({"element", "ui element", "none", "a visible item"})
    __SWIPE_ACTIONS = {"swipe_up", "swipe_down", "swipe_left", "swipe_right", "scroll"}

    __SCREEN_RE = re.compile(
        pattern=r"(?:the\s+)?(\w+(?:\s+\w+)?)\s+(screen|page)\b",
        flags=re.IGNORECASE,
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

    __SCROLL_VERB_RE = re.compile(
        pattern=r"(?:find|look(?:ing)?\s+for|search(?:ing)?\s+for)\s+(.+?)(?:\.|,\s|;|$)",
        flags=re.IGNORECASE,
    )
    __PROPER_PHRASE_RE = re.compile(pattern=r"\b([A-Z][a-z]+(?:\s+[a-z]+)*(?:\s+[A-Z][a-z]+)+)")

    @staticmethod
    def __normalize_positional(target: str) -> str:
        """
        Standardize ordinal formatting in positional target descriptions.
        """

        if not target:
            return target

        text = target.strip()

        def __replace_numeric(match: "re.Match[str]") -> str:
            full = match.group(0).lower()
            return ScriptExporter.__ORDINAL_MAP.get(full, full)

        normalized = ScriptExporter.__NUMERIC_ORDINAL_RE.sub(repl=__replace_numeric, string=text)

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

    @staticmethod
    def __is_intent_target(target: str, intent: str) -> bool:
        """
        Check if a target was mentioned or implied by the user's intent.
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

    @staticmethod
    def __extract_goal_label(goal_state: str) -> str:
        """
        Derive a concise validation label from a potentially long intent.
        """

        if not goal_state:
            return ""

        trimmed = goal_state.strip().rstrip(".")
        if len(trimmed) <= 60 and "." not in trimmed:
            return trimmed

        matches = ScriptExporter.__SCREEN_RE.findall(string=goal_state)

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
        """

        if isinstance(step, StepResult):
            if step.generalized_target:
                return ScriptExporter.__normalize_positional(target=step.generalized_target)
            target = step.step.action.natural_language_target or step.step.action.target
        else:
            if step.get("generalized_target"):
                raw = str(object=step.get("generalized_target") or "")
                return ScriptExporter.__normalize_positional(target=raw)
            target = step.get("natural_language_target") or step.get("target") or ""

        return target or "element"

    @staticmethod
    def __get_event_type(step: Union[StepResult, Dict[str, Any]]) -> str:
        """
        Extract semantic event type from a step.
        """

        if isinstance(step, StepResult):
            return getattr(step.step, "event_type", "action") or "action"

        return str(object=step.get("event_type", "action") or "action")

    @staticmethod
    def __get_action_type(step: Union[StepResult, Dict[str, Any]]) -> str:
        """
        Extract the action type string from a step.
        """

        if isinstance(step, StepResult):
            return step.step.action.action_type.value

        return str(object=step.get("action_type", "unknown"))

    @staticmethod
    def __swipe_direction_label(action_type: str) -> str:
        """
        Map a swipe action to its user-facing scroll/swipe label.
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
        Extract the activity string from a step.
        """

        if isinstance(step, dict):
            return str(object=step.get("activity") or "")

        return ""

    @staticmethod
    def __find_app_launch_boundary(
        steps: Sequence[Union[StepResult, Dict[str, Any]]],
        package_name: str,
    ) -> int:
        """
        Find the index of the first step that runs inside the target package.
        """

        max_launch_steps = 10
        prefix = package_name + "/"

        for j, step in enumerate(iterable=steps):
            if j > max_launch_steps:
                return 0

            activity = ScriptExporter.__get_activity(step=step)
            if activity.startswith(prefix) or activity == package_name:
                return j

        return 0

    @staticmethod
    def export(
        step_results: Sequence[Union[StepResult, Dict[str, Any]]],
        *,
        intent: str = "",
        goal_state: str = "",
        package_name: str = "",
    ) -> str:
        """
        Export steps to a natural language test script.
        """

        lines: list[str] = []

        # Filter recovery steps
        filtered_results = []
        for step in step_results:
            if ScriptExporter.__get_condition(step=step) == "recovery":
                continue
            filtered_results.append(step)

        step_results = filtered_results

        n = len(step_results)
        i = 0
        launch_boundary = 0
        swipe_just_processed = False
        validation_subjects = ScriptExporter.__extract_validation_subjects(
            intent=(intent or goal_state)
        )
        validation_subject_index = 0
        emitted_validation_lines: set[str] = set()

        if package_name:
            launch_boundary = ScriptExporter.__find_app_launch_boundary(
                steps=step_results, package_name=package_name
            )
            if launch_boundary > 0:
                lines.append(f"OPEN_APP {package_name}")
                i = launch_boundary

        while i < n:
            step = step_results[i]
            action_type_val = ScriptExporter.__get_action_type(step=step)
            event_type = ScriptExporter.__get_event_type(step=step)
            condition = ScriptExporter.__get_condition(step=step)
            target = ScriptExporter.__resolve_target(step=step)

            if action_type_val in ScriptExporter.__SWIPE_ACTIONS:
                swipe_direction = action_type_val
                swipe_start = i
                j = i + 1
                while (
                    j < n
                    and ScriptExporter.__get_action_type(step=step_results[j]) == swipe_direction
                ):
                    j += 1
                i = j

                if i < n:
                    next_target = ScriptExporter.__resolve_target(step=step_results[i])
                else:
                    # Advanced Scroll Inference (Restored)
                    next_target = (
                        ScriptExporter.__infer_scroll_target(
                            steps=step_results, start=swipe_start, end=i
                        )
                        or ScriptExporter.__extract_goal_label(goal_state=(intent or goal_state))
                        or intent
                        or goal_state
                        or "the target"
                    )

                label = ScriptExporter.__swipe_direction_label(action_type=swipe_direction)
                lines.append(f"{label} until {next_target} is visible")
                swipe_just_processed = True
                continue

            # Smart Validation on screen change (Restored)
            if (
                i > 0
                and i > launch_boundary
                and action_type_val != "wait"
                and not swipe_just_processed
                and event_type != "validation"
                and not condition
            ):
                prev = step_results[i - 1]
                prev_action_type = ScriptExporter.__get_action_type(step=prev)
                prev_condition = ScriptExporter.__get_condition(step=prev)
                prev_changed = (
                    prev.screen_changed
                    if isinstance(prev, StepResult)
                    else prev.get("screen_changed", False)
                )
                if (
                    prev_changed
                    and prev_action_type not in ("wait", *ScriptExporter.__SWIPE_ACTIONS)
                    and target.lower() not in ScriptExporter.__GENERIC_TARGETS
                ):
                    val_line = f"Validate {target} is visible"
                    if prev_condition:
                        lines.append(f"IF {prev_condition}")
                        lines.append("{")
                        lines.append(f"    {val_line}")
                        lines.append("}")
                    else:
                        lines.append(val_line)

            swipe_just_processed = False

            if isinstance(step, StepResult):
                action = step.step.action
                text = action.text
                rationale = action.rationale
                wait_duration = action.wait_duration
            else:
                text = step.get("text")
                rationale = str(object=step.get("rationale", ""))
                wait_duration = step.get("wait_duration")

            if action_type_val == "wait" and target.lower() in ScriptExporter.__GENERIC_TARGETS:
                target = ScriptExporter.__infer_wait_subject(rationale=rationale)

            description = Normalizer.action(
                action_type=action_type_val, target=target, text=text, wait_duration=wait_duration
            )

            # Semantic Validation Handling (Integrated)
            if event_type == "validation":
                effective_target = target
                is_system_validation = ScriptExporter.__is_system_validation(
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
                validation_condition = ScriptExporter.__infer_validation_condition(
                    condition=condition,
                    action_type=action_type_val,
                    target=effective_target,
                    rationale=rationale,
                )
                if not validation_condition and i > 0 and not should_use_intent_subject:
                    prev = step_results[i - 1]
                    prev_condition = ScriptExporter.__get_condition(step=prev)
                    prev_action_type = ScriptExporter.__get_action_type(step=prev)
                    if prev_condition and prev_action_type == "wait":
                        validation_condition = prev_condition

                validation_line = Normalizer.validation(
                    target=effective_target,
                    explicit=should_use_intent_subject,
                )

                if validation_condition and i + 1 < n:
                    next_step = step_results[i + 1]
                    next_condition = ScriptExporter.__get_condition(step=next_step)
                    next_target = ScriptExporter.__resolve_target(step=next_step)
                    next_event_type = ScriptExporter.__get_event_type(step=next_step)
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
                    merged_into_previous_wait_block = False
                    if i > 0:
                        prev = step_results[i - 1]
                        prev_action_type = ScriptExporter.__get_action_type(step=prev)
                        prev_condition = ScriptExporter.__get_condition(step=prev)
                        if (
                            prev_action_type == "wait"
                            and prev_condition == validation_condition
                            and len(lines) >= 4
                            and lines[-4] == f"IF {validation_condition}"
                            and lines[-3] == "{"
                            and lines[-1] == "}"
                        ):
                            lines.pop()
                            lines.append(f"    {validation_line}")
                            lines.append("}")
                            merged_into_previous_wait_block = True
                    if merged_into_previous_wait_block:
                        emitted_validation_lines.add(validation_line)
                        i += 1
                        continue
                    lines.append(f"IF {validation_condition}")
                    lines.append("{")
                    lines.append(f"    {validation_line}")
                    lines.append("}")
                else:
                    lines.append(validation_line)
                emitted_validation_lines.add(validation_line)
                i += 1
                continue

            if condition:
                lines.append(f"IF {condition}")
                lines.append("{")
                prev_is_same_target_validation = False
                if i > 0:
                    prev = step_results[i - 1]
                    prev_event_type = ScriptExporter.__get_event_type(step=prev)
                    prev_target = ScriptExporter.__resolve_target(step=prev)
                    prev_is_same_target_validation = (
                        prev_event_type == "validation"
                        and prev_target.strip().lower() == target.strip().lower()
                    )
                if (
                    target.lower() not in ScriptExporter.__GENERIC_TARGETS
                    and action_type_val != "wait"
                    and not prev_is_same_target_validation
                ):
                    lines.append(f"    Validate {target} is visible")
                lines.append(f"    {description}")
                lines.append("}")
            else:
                lines.append(description)
            i += 1

        # Final Goal Validation Logic (Restored)
        if step_results:
            last_action_type = ScriptExporter.__get_action_type(step=step_results[-1])

            if last_action_type not in ("complete", "verify_goal_completion"):
                if validation_subjects:
                    final_validation_line = f"Validate that {validation_subjects[-1]}"
                    if final_validation_line not in emitted_validation_lines and (
                        not lines or lines[-1].strip() != final_validation_line
                    ):
                        lines.append(final_validation_line)
                        emitted_validation_lines.add(final_validation_line)
                    return "\n".join(lines) + "\n"

                goal_label = ScriptExporter.__extract_goal_label(goal_state=(intent or goal_state))
                last_target = ScriptExporter.__resolve_target(step=step_results[-1])
                last_target_usable = (
                    last_target and last_target.lower() not in ScriptExporter.__GENERIC_TARGETS
                )

                if last_target_usable and ScriptExporter.__is_intent_target(
                    target=last_target, intent=(intent or goal_state)
                ):
                    val_line = f"Validate {last_target} is visible"
                    if not lines or lines[-1].strip() != val_line:
                        lines.append(val_line)
                elif goal_label:
                    val_line = f"Validate {goal_label} is visible"
                    if not lines or lines[-1].strip() != val_line:
                        lines.append(val_line)
                elif last_target_usable:
                    val_line = f"Validate {last_target} is visible"
                    if not lines or lines[-1].strip() != val_line:
                        lines.append(val_line)
                else:
                    val_line = "Validate Goal State is visible"
                    if not lines or lines[-1].strip() != val_line:
                        lines.append(val_line)

        return "\n".join(lines) + "\n"

    @staticmethod
    def __is_system_validation(
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
    def __extract_validation_subjects(intent: str) -> list[str]:
        """
        Extract all user-requested validation subjects from intent text.
        """

        if not intent:
            return []

        matches = re.finditer(
            pattern=r"\b(?:validate|verify|check|confirm)(?:\s+that)?\s+(.+?)(?=(?:,|\bthen\b|\band\s+(?:validate|verify|check|confirm)\b|$))",
            string=intent,
            flags=re.IGNORECASE,
        )
        subjects: list[str] = []
        for match in matches:
            subject = Normalizer.clean(text=match.group(1).strip(" .,:;"))
            if subject:
                subjects.append(subject)

        return subjects

    @staticmethod
    def __infer_scroll_target(
        steps: Sequence[Union[StepResult, Dict[str, Any]]],
        start: int,
        end: int,
    ) -> str:
        """
        Infer what the user was scrolling to find from swipe rationales.
        """

        for j in range(start, min(end, start + 5)):
            step = steps[j]
            if isinstance(step, StepResult):
                rationale = step.step.action.rationale or ""
            else:
                rationale = str(object=step.get("rationale") or "")
            if not rationale:
                continue
            verb_match = ScriptExporter.__SCROLL_VERB_RE.search(string=rationale)
            if not verb_match:
                continue
            clause = verb_match.group(1)
            product_match = ScriptExporter.__PROPER_PHRASE_RE.search(string=clause)
            if product_match:
                return product_match.group(1).strip()
        return ""

    @staticmethod
    def __infer_wait_subject(rationale: Optional[str]) -> str:
        """
        Derive a human-readable wait subject from the step rationale.
        """

        return Normalizer.wait_subject(rationale=rationale) or "screen to load"

    @staticmethod
    def __infer_validation_condition(
        *,
        target: str,
        action_type: str,
        rationale: Optional[str],
        condition: Optional[str],
    ) -> Optional[str]:
        """
        Infer IF conditions for validation events.
        """

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
            if target.lower() in ScriptExporter.__GENERIC_TARGETS:
                return f"{ScriptExporter.__infer_wait_subject(rationale=rationale)} is visible"

            return f"{target} is visible"

        return None

    @staticmethod
    def __get_condition(step: Union[StepResult, Dict[str, Any]]) -> Optional[str]:
        """
        Get the condition for a step, inferring from rationale if needed.
        """

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
            if "timeout" in lower_rationale:
                condition = "Timeout error is displayed"
            elif (
                "retry" in lower_rationale
                or "try again" in lower_rationale
                or "error" in lower_rationale
            ):
                condition = "Error message is displayed"

        if action_type == "wait" and not condition:
            resolved = ScriptExporter.__resolve_target(step=step)
            if resolved.lower() in ScriptExporter.__GENERIC_TARGETS:
                subject = ScriptExporter.__infer_wait_subject(rationale=rationale)
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
