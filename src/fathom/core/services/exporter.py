from __future__ import annotations

import re
from typing import Any, Dict, Literal, Optional, Sequence, Union, cast

from fathom.core.exceptions import ScriptExportError
from fathom.core.prompts.export import ExportPromptBuilder
from fathom.core.prompts.factory import PromptFactory
from fathom.core.prompts.tools import ToolRegistry
from fathom.core.services.normalizer import Normalizer
from fathom.interfaces.llm import LLMPort
from fathom.schemas.export import ScriptExportPayload, ScriptExportStructuredPayload
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
    __DYNAMIC_TARGET_PREFIXES = (
        "add to cart button for ",
        "increase quantity button for ",
        "decrease quantity button for ",
        "remove item button for ",
    )
    __STORE_NAME_PATTERN = re.compile(
        pattern=(
            r"\b(?:"
            r"walmart|costco|target|kroger|safeway|publix|aldi|instacart|"
            r"whole\s+foods|trader\s+joe'?s|amazon\s+fresh|tesco"
            r")\b\s+"
            r"(?=(?:"
            r"continue\s+shopping|cart|button|item|entry|row|store|aisle|checkout|basket"
            r")\b)"
        ),
        flags=re.IGNORECASE,
    )

    def __init__(self, *, llm: Optional[LLMPort] = None, use_cache: bool = True) -> None:
        """
        Initialize exporter with optional LLM-backed script composition.
        """

        self.__llm = llm
        self.__use_cache = use_cache
        self.__prompt_builder: Optional[ExportPromptBuilder] = (
            PromptFactory.get_export_builder(model_name=llm.model_name) if llm else None
        )

    @staticmethod
    def __normalize_script_output(script: str) -> str:
        """
        Normalize generated script formatting and ensure trailing newline.
        """

        cleaned_lines = [line.rstrip() for line in str(script).replace("\r\n", "\n").split("\n")]
        while cleaned_lines and not cleaned_lines[-1].strip():
            cleaned_lines.pop()
        if not cleaned_lines:
            return ""
        return "\n".join(cleaned_lines) + "\n"

    @staticmethod
    def __normalize_text_signal(text: str) -> str:
        """
        Normalize text for fuzzy containment checks.
        """

        cleaned = re.sub(pattern=r"[^a-z0-9\s]", repl=" ", string=str(text).lower())
        return re.sub(pattern=r"\s+", repl=" ", string=cleaned).strip()

    @staticmethod
    def __intent_mentions_phrase(intent: str, phrase: str) -> bool:
        """
        Check if a phrase is explicitly present in user intent.
        """

        if not intent or not phrase:
            return False

        intent_norm = ScriptExporter.__normalize_text_signal(text=intent)
        phrase_norm = ScriptExporter.__normalize_text_signal(text=phrase)
        return bool(phrase_norm) and phrase_norm in intent_norm

    @staticmethod
    def __is_generic_dynamic_reference(text: str) -> bool:
        """
        Detect whether a dynamic target is already generic/repeatable.
        """

        lower = str(text).strip().lower()
        generic_tokens = (
            "first",
            "second",
            "third",
            "search result",
            "matching result",
            "matching item",
            "selected item",
        )
        return any(token in lower for token in generic_tokens)

    @staticmethod
    def __generalize_dynamic_target(target: str, intent: str, *, generic_item_phrase: str) -> str:
        """
        Generalize dynamic product-specific targets unless explicitly requested by intent.
        """

        if not target:
            return target

        lowered = target.lower()
        for prefix in ScriptExporter.__DYNAMIC_TARGET_PREFIXES:
            marker = lowered.find(prefix)
            if marker < 0:
                continue

            suffix_start = marker + len(prefix)
            specific_phrase = target[suffix_start:].strip()
            if not specific_phrase:
                return target
            if ScriptExporter.__intent_mentions_phrase(intent=intent, phrase=specific_phrase):
                return target
            if ScriptExporter.__is_generic_dynamic_reference(text=specific_phrase):
                return target

            return f"{target[:suffix_start]}{generic_item_phrase}"

        return target

    @staticmethod
    def __sanitize_script_targets(script: str, intent: str) -> str:
        """
        Post-process script lines to improve repeatability of dynamic targets.
        """

        if not script:
            return script

        lines = script.splitlines()
        updated: list[str] = []
        combined_signal = ScriptExporter.__normalize_text_signal(text=f"{intent} {script}")
        search_context = any(
            token in combined_signal
            for token in ("search bar", "search suggestion", "search result", "search")
        )
        generic_item_phrase = (
            "the first search result" if search_context else "the first matching item"
        )

        for line in lines:
            transformed = ScriptExporter.__STORE_NAME_PATTERN.sub(repl="", string=line)
            for prefix in ScriptExporter.__DYNAMIC_TARGET_PREFIXES:
                pattern = re.compile(
                    pattern=rf"({re.escape(prefix)})(.+?)(?=(\s+is\s+visible\b|$))",
                    flags=re.IGNORECASE,
                )

                def __replace(match: "re.Match[str]") -> str:
                    left = match.group(1)
                    suffix = match.group(2).strip()
                    combined = f"{left}{suffix}"
                    generalized = ScriptExporter.__generalize_dynamic_target(
                        target=combined, intent=intent, generic_item_phrase=generic_item_phrase
                    )
                    if not generalized.lower().startswith(left.lower()):
                        return match.group(0)
                    return generalized

                transformed = pattern.sub(repl=__replace, string=transformed)
            transformed = re.sub(pattern=r"\s{2,}", repl=" ", string=transformed).strip()
            updated.append(transformed)

        normalized = "\n".join(updated)
        return ScriptExporter.__normalize_script_output(script=normalized)

    @staticmethod
    def __count_action_lines(script: str) -> int:
        """
        Count non-structural lines to prevent empty/underspecified LLM scripts.
        """

        count = 0
        for line in script.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped == "}" or stripped.startswith("IF ") and stripped.endswith("{"):
                continue
            count += 1
        return count

    @staticmethod
    def __action_kind_from_line(line: str) -> Optional[str]:
        """
        Extract canonical executable action kind from a script line.
        """

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
            return "swipe"
        if normalized.startswith("wait "):
            return "wait"
        if normalized.startswith("press "):
            return "press"
        if normalized.startswith("long press "):
            return "long_press"
        return None

    @staticmethod
    def __executable_action_counts(script: str) -> Dict[str, int]:
        """
        Count executable action lines (excluding validations and IF structure).
        """

        counts: Dict[str, int] = {}
        for raw_line in script.splitlines():
            line = raw_line.strip()
            if not line or line == "}":
                continue
            if line.startswith("IF ") and line.endswith("{"):
                continue

            action_kind = ScriptExporter.__action_kind_from_line(line=line)
            if not action_kind:
                continue
            counts[action_kind] = counts.get(action_kind, 0) + 1
        return counts

    @staticmethod
    def __is_valid_llm_script(candidate: str, baseline: str) -> bool:
        """
        Validate LLM output is a plain script and structurally safe.
        """

        if not candidate.strip():
            return False
        if "```" in candidate:
            return False

        balance = 0
        for raw_line in candidate.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("IF ") and line.endswith("{"):
                balance += 1
                continue
            if line == "}":
                balance -= 1
                if balance < 0:
                    return False
        if balance != 0:
            return False

        baseline_actions = ScriptExporter.__count_action_lines(script=baseline)
        candidate_actions = ScriptExporter.__count_action_lines(script=candidate)
        if baseline_actions > 0 and candidate_actions <= 0:
            return False

        baseline_counts = ScriptExporter.__executable_action_counts(script=baseline)
        candidate_counts = ScriptExporter.__executable_action_counts(script=candidate)
        for action_kind, required_count in baseline_counts.items():
            if candidate_counts.get(action_kind, 0) < required_count:
                return False

        return True

    @staticmethod
    def __last_non_structural_line(script: str) -> str:
        """
        Return the last meaningful script line excluding IF/brace structure.
        """

        for raw_line in reversed(script.splitlines()):
            line = raw_line.strip()
            if not line:
                continue
            if line == "}":
                continue
            if line.startswith("IF ") and line.endswith("{"):
                continue
            return line
        return ""

    @staticmethod
    def __contains_goal_validation(script: str) -> bool:
        """
        Ensure the script ends with an explicit validation statement.
        """

        last_line = ScriptExporter.__last_non_structural_line(script=script)
        if not last_line:
            return False
        return last_line.lower().startswith("validate")

    @staticmethod
    def __normalize_structured_action_ids(
        structured_args: Dict[str, Any], required_action_ids: Sequence[str]
    ) -> Dict[str, Any]:
        """
        Normalize Gemini structured action IDs to include all required IDs exactly once.
        Keeps model-provided conditional grouping, appends missing IDs to remaining_action_ids.
        """

        normalized = dict(structured_args)
        conditional_blocks_raw = list(normalized.get("conditional_blocks") or [])
        remaining_raw = list(normalized.get("remaining_action_ids") or [])
        required_set = set(required_action_ids)

        seen: set[str] = set()
        cleaned_blocks: list[Dict[str, Any]] = []
        for block in conditional_blocks_raw:
            if not isinstance(block, dict):
                continue
            condition = str(block.get("condition") or "").strip()
            action_ids_raw = block.get("action_ids") or []
            block_ids: list[str] = []
            for action_id in action_ids_raw:
                aid = str(action_id).strip()
                if not aid or aid in seen or aid not in required_set:
                    continue
                seen.add(aid)
                block_ids.append(aid)
            cleaned_blocks.append({"condition": condition, "action_ids": block_ids})

        cleaned_remaining: list[str] = []
        for action_id in remaining_raw:
            aid = str(action_id).strip()
            if not aid or aid in seen or aid not in required_set:
                continue
            seen.add(aid)
            cleaned_remaining.append(aid)

        for required_id in required_action_ids:
            if required_id not in seen:
                cleaned_remaining.append(required_id)
                seen.add(required_id)

        normalized["conditional_blocks"] = cleaned_blocks
        normalized["remaining_action_ids"] = cleaned_remaining
        return normalized

    @staticmethod
    def __intent_requires_if_block(intent: str) -> bool:
        """
        Determine whether intent explicitly requests conditional flow.
        """

        normalized = ScriptExporter.__normalize_text_signal(text=intent)
        if not normalized:
            return False

        conditional_terms = ("if ", " when ", "if the", "if cart", "if there are")
        if not any(term in f" {normalized} " for term in conditional_terms):
            return False

        domain_terms = ("cart", "not empty", "has any item", "contains item", "clear")
        return any(term in normalized for term in domain_terms)

    @staticmethod
    def __build_export_payload(
        step_results: Sequence[Union[StepResult, Dict[str, Any]]],
    ) -> list[Dict[str, Any]]:
        """
        Build compact structured payload from step history for LLM export.
        """

        payload: list[Dict[str, Any]] = []
        for index, step in enumerate(step_results, start=1):
            action_type_val = ScriptExporter.__get_action_type(step=step)
            target = ScriptExporter.__resolve_target(step=step)
            condition = ScriptExporter.__get_condition(step=step)
            event_type = ScriptExporter.__get_event_type(step=step)
            is_conditional = ScriptExporter.__is_explicit_conditional(step=step)
            conditional_type = ScriptExporter.__get_conditional_type(step=step)

            if isinstance(step, StepResult):
                text = step.step.action.text
                rationale = step.step.action.rationale
                screen_changed = bool(step.screen_changed)
            else:
                text = step.get("text")
                rationale = str(object=step.get("rationale") or "")
                screen_changed = bool(step.get("screen_changed", False))

            payload.append(
                {
                    "step": index,
                    "event_type": event_type,
                    "action_type": action_type_val,
                    "target": target,
                    "text": text,
                    "condition": condition,
                    "is_conditional": is_conditional,
                    "conditional_type": conditional_type,
                    "screen_changed": screen_changed,
                    "rationale": rationale,
                }
            )
        return payload

    async def export_with_llm(
        self,
        step_results: Sequence[Union[StepResult, Dict[str, Any]]],
        *,
        intent: str = "",
        goal_state: str = "",
        package_name: str = "",
    ) -> Optional[str]:
        """
        Export script using Gemini-only composition.
        """

        if not self.__llm:
            return None

        payload = ScriptExporter.__build_export_payload(step_results=step_results)
        if not payload:
            return ""

        if not self.__prompt_builder:
            raise ScriptExportError("Script exporter prompt builder is not configured.")

        llm_baseline_script = self.export(
            step_results=step_results,
            intent=intent,
            goal_state=goal_state,
            package_name=package_name,
            include_final_validation=False,
        )
        llm_baseline_script = ScriptExporter.__normalize_script_output(script=llm_baseline_script)
        llm_baseline_script = ScriptExporter.__sanitize_script_targets(
            script=llm_baseline_script, intent=(intent or goal_state)
        )
        action_catalog, required_action_ids, required_open_app_id = (
            ScriptExporter.__build_action_catalog_from_steps(
                step_results=step_results,
                package_name=package_name,
                intent=(intent or goal_state),
            )
        )
        action_catalog_lines = [
            f"{action_id}: {line}" for action_id, line in action_catalog.items()
        ]
        allowed_action_lines = [
            line.strip().lower() for line in action_catalog.values() if line.strip()
        ]
        strict_enforcement_ready = len(step_results) >= 4

        system_instruction = self.__prompt_builder.build_system_instruction()
        prompt_text = self.__prompt_builder.build_user_prompt(
            intent=intent,
            goal_state=goal_state,
            package_name=package_name,
            baseline_script=llm_baseline_script,
            trace_payload=payload,
            action_catalog_lines=action_catalog_lines,
        )

        try:
            response = await self.__llm.generate(
                use_cache=self.__use_cache,
                prompt=[prompt_text],
                tools=ToolRegistry.get_export_definitions(),
                system_instruction=system_instruction,
            )
        except Exception as exception:
            raise ScriptExportError(f"Gemini script generation failed: {exception}") from exception

        structured_args: Dict[str, Any] = {}
        if response.tool_calls:
            for tool_call in response.tool_calls:
                if getattr(tool_call, "name", "") != "emit_script":
                    continue
                arguments = getattr(tool_call, "args", {}) or {}
                structured_args = dict(arguments)
                break

        if not structured_args:
            raise ScriptExportError("Gemini did not return structured emit_script tool arguments.")

        normalized_structured_args = ScriptExporter.__normalize_structured_action_ids(
            structured_args=structured_args, required_action_ids=required_action_ids
        )

        try:
            structured_payload = ScriptExportStructuredPayload.model_validate(
                {
                    **normalized_structured_args,
                    "action_catalog": action_catalog,
                    "required_action_ids": required_action_ids if strict_enforcement_ready else [],
                    "required_open_app_id": required_open_app_id
                    if strict_enforcement_ready
                    else None,
                    "require_if_block": (
                        ScriptExporter.__intent_requires_if_block(intent=(intent or goal_state))
                        if strict_enforcement_ready
                        else False
                    ),
                }
            )
        except Exception as exception:
            raise ScriptExportError(
                f"Gemini-generated structured payload failed schema validation: {exception}"
            ) from exception

        candidate = ScriptExporter.__normalize_script_output(script=structured_payload.to_script())
        candidate = ScriptExporter.__sanitize_script_targets(
            script=candidate, intent=(intent or goal_state)
        )

        try:
            parsed_script = ScriptExportPayload.model_validate(
                {
                    "script": candidate,
                    "allowed_action_lines": allowed_action_lines
                    if strict_enforcement_ready
                    else [],
                    "required_open_app": (
                        f"open_app {package_name}".strip()
                        if package_name and strict_enforcement_ready
                        else None
                    ),
                    "require_if_block": (
                        ScriptExporter.__intent_requires_if_block(intent=(intent or goal_state))
                        if strict_enforcement_ready
                        else False
                    ),
                }
            )
        except Exception as exception:
            raise ScriptExportError(
                f"Gemini-generated script failed Pydantic schema validation: {exception}"
            ) from exception

        candidate = parsed_script.script
        if strict_enforcement_ready:
            if not ScriptExporter.__is_valid_llm_script(
                candidate=candidate, baseline=llm_baseline_script
            ):
                raise ScriptExportError(
                    "Gemini-generated script failed structural or action-coverage validation."
                )
            if not ScriptExporter.__contains_goal_validation(script=candidate):
                raise ScriptExportError(
                    "Gemini-generated script does not contain a valid final goal-validation step."
                )

        return candidate

    @staticmethod
    def __build_action_catalog_from_steps(
        step_results: Sequence[Union[StepResult, Dict[str, Any]]],
        package_name: str,
        intent: str,
    ) -> tuple[Dict[str, str], list[str], Optional[str]]:
        """
        Build ordered executable action catalog from raw step data.
        """

        lines: list[str] = []

        if package_name:
            lines.append(f"OPEN_APP {package_name}")

        for step in step_results:
            action_type_val = ScriptExporter.__get_action_type(step=step)
            target = ScriptExporter.__resolve_target(step=step)

            if isinstance(step, StepResult):
                text = step.step.action.text
                rationale = step.step.action.rationale
                wait_duration = step.step.action.wait_duration
            else:
                text = step.get("text")
                rationale = str(object=step.get("rationale", ""))
                wait_duration = step.get("wait_duration")

            if action_type_val == "wait" and target.lower() in ScriptExporter.__GENERIC_TARGETS:
                target = ScriptExporter.__infer_wait_subject(rationale=rationale)

            description = Normalizer.action(
                action_type=action_type_val, target=target, text=text, wait_duration=wait_duration
            )

            lowered = description.lower()
            if lowered.startswith("validate "):
                continue

            # Collapse app launch into OPEN_APP by skipping the initial app-icon tap.
            is_launch_tap = (
                bool(package_name)
                and not lines[1:]  # OPEN_APP is already present; this is first step-derived action
                and action_type_val == "tap"
                and "app icon" in target.lower()
            )
            if is_launch_tap:
                continue

            lines.append(description)

        executable_prefixes = (
            "open_app ",
            "tap ",
            "type ",
            "scroll ",
            "swipe ",
            "wait ",
            "press ",
            "long press ",
        )
        sanitized_script = ScriptExporter.__sanitize_script_targets(
            script="\n".join(lines) + ("\n" if lines else ""),
            intent=intent,
        )
        executable_lines = [
            line.strip()
            for line in sanitized_script.splitlines()
            if line.strip() and line.strip().lower().startswith(executable_prefixes)
        ]

        action_catalog: Dict[str, str] = {}
        required_action_ids: list[str] = []
        required_open_app_id: Optional[str] = None

        for index, line in enumerate(executable_lines, start=1):
            action_id = f"A{index}"
            action_catalog[action_id] = line
            required_action_ids.append(action_id)
            if required_open_app_id is None and line.lower().startswith("open_app "):
                required_open_app_id = action_id

        return action_catalog, required_action_ids, required_open_app_id

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
    def __is_overlay_detected(step: Union[StepResult, Dict[str, Any]]) -> bool:
        """
        Extract explicit overlay/popup blocker signal from a step.
        """

        if isinstance(step, StepResult):
            return bool(getattr(step.step.action, "overlay_detected", False))

        return bool(step.get("overlay_detected", False))

    @staticmethod
    def __is_explicit_conditional(step: Union[StepResult, Dict[str, Any]]) -> bool:
        """
        Extract explicit conditional execution signal from a step.
        """

        if isinstance(step, StepResult):
            return bool(getattr(step.step.action, "is_conditional", False))

        return bool(step.get("is_conditional", False))

    @staticmethod
    def __get_conditional_type(
        step: Union[StepResult, Dict[str, Any]],
    ) -> Optional[Literal["blocker", "transient", "error", "optional"]]:
        """
        Extract conditional type for explicit conditional actions.
        """

        if isinstance(step, StepResult):
            raw = getattr(step.step.action, "conditional_type", None)
        else:
            raw = step.get("conditional_type")

        text = str(raw or "").strip().lower()
        if text in ("blocker", "transient", "error", "optional"):
            return cast("Literal['blocker', 'transient', 'error', 'optional']", text)
        return None

    @staticmethod
    def __default_condition_for_type(
        conditional_type: Optional[Literal["blocker", "transient", "error", "optional"]],
    ) -> Optional[str]:
        """
        Map explicit conditional type to deterministic default condition text.
        """

        mapping = {
            "blocker": "Blocker prompt is visible",
            "transient": "Transient screen is visible",
            "error": "Error message is displayed",
            "optional": "Optional UI state is visible",
        }
        return mapping.get(conditional_type or "")

    @staticmethod
    def __is_generic_wait_condition(condition: Optional[str]) -> bool:
        """
        Detect generic wait-derived conditions that should be replaced by explicit conditional type defaults.
        """

        if not condition:
            return False

        lower = condition.strip().lower()
        generic_wait_phrases = {
            "screen to load is visible",
            "the app is still loading",
            "loading spinner is visible",
        }
        return lower in generic_wait_phrases

    @staticmethod
    def __get_raw_condition(step: Union[StepResult, Dict[str, Any]]) -> Optional[str]:
        """
        Extract condition directly from step payload without heuristic inference.
        """

        if isinstance(step, StepResult):
            raw = getattr(step.step, "condition", None) or getattr(
                step.step.action, "condition", None
            )
        else:
            raw = step.get("condition")

        text = str(raw).strip() if raw else None
        return text or None

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
    def __infer_open_app_package(
        steps: Sequence[Union[StepResult, Dict[str, Any]]],
        default_package: str,
    ) -> Optional[str]:
        """
        Infer package name for OPEN_APP from step activity when available.
        Falls back to the provided package identifier.
        """

        if default_package:
            return default_package

        if not steps:
            return None

        # Try to find the first concrete activity and derive package from it.
        for step in steps:
            activity = ScriptExporter.__get_activity(step=step)
            if not activity:
                continue
            if "/" in activity:
                return activity.split("/")[0].strip() or None
            return activity.strip() or None

        return None

    @staticmethod
    def export(
        step_results: Sequence[Union[StepResult, Dict[str, Any]]],
        *,
        intent: str = "",
        goal_state: str = "",
        package_name: str = "",
        include_final_validation: bool = True,
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
            else:
                inferred_package = ScriptExporter.__infer_open_app_package(
                    steps=step_results, default_package=package_name
                )
                if inferred_package:
                    lines.append(f"OPEN_APP {inferred_package}")
                    i = 1

        while i < n:
            step = step_results[i]
            action_type_val = ScriptExporter.__get_action_type(step=step)
            event_type = ScriptExporter.__get_event_type(step=step)
            raw_condition = ScriptExporter.__get_raw_condition(step=step)
            condition = ScriptExporter.__get_condition(step=step)
            explicit_conditional = ScriptExporter.__is_explicit_conditional(step=step)
            conditional_type = ScriptExporter.__get_conditional_type(step=step)
            if explicit_conditional:
                default_condition = ScriptExporter.__default_condition_for_type(
                    conditional_type=conditional_type
                )
                if default_condition and (
                    not raw_condition
                    or not condition
                    or ScriptExporter.__is_generic_wait_condition(condition=condition)
                ):
                    condition = default_condition
            if ScriptExporter.__is_overlay_detected(
                step=step
            ) and not ScriptExporter.__is_blocker_popup_condition(condition=condition):
                condition = "Overlay is visible"
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
                        lines.append(f"IF {prev_condition} {{ {val_line} }}")
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
                            and len(lines) >= 3
                            and lines[-3] == f"IF {validation_condition} {{"
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
                    lines.append(f"IF {validation_condition} {{ {validation_line} }}")
                else:
                    lines.append(validation_line)
                emitted_validation_lines.add(validation_line)
                i += 1
                continue

            if condition and action_type_val != "wait":
                lines.append(f"IF {condition} {{")
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
        if include_final_validation and step_results:
            last_action_type = ScriptExporter.__get_action_type(step=step_results[-1])

            if last_action_type not in ("complete", "verify_goal_completion"):
                if validation_subjects:
                    final_validation_line = f"Validate that {validation_subjects[-1]}"
                    if final_validation_line not in emitted_validation_lines and (
                        not lines or lines[-1].strip() != final_validation_line
                    ):
                        lines.append(final_validation_line)
                        emitted_validation_lines.add(final_validation_line)
                    script = "\n".join(lines) + "\n"
                    return ScriptExporter.__sanitize_script_targets(
                        script=script, intent=(intent or goal_state)
                    )

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

        script = "\n".join(lines) + "\n"
        return ScriptExporter.__sanitize_script_targets(
            script=script, intent=(intent or goal_state)
        )

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
    def __is_blocker_popup_condition(condition: Optional[str]) -> bool:
        """
        Check whether a condition corresponds to blocker/popup style UI states.
        """

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
