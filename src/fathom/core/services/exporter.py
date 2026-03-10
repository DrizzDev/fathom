from __future__ import annotations

import json
import re
from logging import getLogger
from typing import Any, Dict, Literal, Optional, Sequence, Union, cast

from fathom.constants.execution import LAUNCHER_PACKAGES
from fathom.core.exceptions import ScriptExportError
from fathom.core.prompts.export import ExportPromptBuilder
from fathom.core.prompts.factory import PromptFactory
from fathom.core.prompts.tools import ToolRegistry
from fathom.core.services.normalizer import Normalizer
from fathom.interfaces.llm import LLMPort
from fathom.schemas.export import ScriptExportPayload, ScriptExportStructuredPayload
from fathom.schemas.gemini_tools import EmitScriptArgs
from fathom.schemas.steps import StepResult

logger = getLogger(__name__)


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
    __GENERIC_TARGETS = frozenset(
        {
            # Shared generic target names plus exporter-specific variants.
            "element",
            "ui element",
            "none",
            "label",
            "unknown",
            "a visible item",
        }
    )
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
            # Coverage validation treats swipe/scroll as the same navigation family.
            return "scroll"
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
        return not (baseline_actions > 0 and candidate_actions <= 0)

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
        structured_args: Dict[str, Any],
        required_action_ids: Sequence[str],
        action_catalog: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Normalize Gemini structured action IDs to include all required IDs exactly once.
        Keeps model-provided conditional grouping, appends missing IDs to remaining_action_ids.
        """

        normalized = dict(structured_args)
        normalized["final_validation"] = ScriptExporter.__normalize_final_validation(
            value=normalized.get("final_validation")
        )
        raw_action_validations = normalized.get("action_validations")
        normalized_action_validations: Dict[str, str] = {}
        if isinstance(raw_action_validations, dict):
            for action_id, validation_text in raw_action_validations.items():
                aid = str(action_id).strip()
                if not aid:
                    continue
                normalized_action_validations[aid] = ScriptExporter.__normalize_validation_line(
                    value=validation_text,
                    fallback="Validate expected state is visible.",
                )
        normalized["action_validations"] = normalized_action_validations
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

        action_catalog = action_catalog or {}
        order_rank = {action_id: idx for idx, action_id in enumerate(required_action_ids)}
        moved_to_blocks: list[str] = []
        for block in cleaned_blocks:
            condition_lower = str(block.get("condition") or "").strip().lower()
            block_ids = list(block.get("action_ids") or [])
            if "outside the us dropdown" not in condition_lower or not block_ids:
                continue

            last_rank = max(order_rank.get(action_id, -1) for action_id in block_ids)
            candidate_ids: list[str] = []
            for action_id in cleaned_remaining:
                rank = order_rank.get(action_id, -1)
                if rank <= last_rank:
                    continue
                action_line = str(action_catalog.get(action_id) or "").strip().lower()
                is_scroll = action_line.startswith("scroll ")
                is_location_selection_tap = action_line.startswith("tap on ") and (
                    "washington" in action_line or action_line.endswith(" option")
                )
                if is_scroll or is_location_selection_tap:
                    candidate_ids.append(action_id)
                    last_rank = rank
                    continue
                if candidate_ids:
                    break

            if candidate_ids:
                block["action_ids"] = block_ids + candidate_ids
                moved_to_blocks.extend(candidate_ids)

        if moved_to_blocks:
            moved_set = set(moved_to_blocks)
            cleaned_remaining = [
                action_id for action_id in cleaned_remaining if action_id not in moved_set
            ]

        normalized["conditional_blocks"] = cleaned_blocks
        normalized["remaining_action_ids"] = cleaned_remaining
        return normalized

    @staticmethod
    def __normalize_final_validation(value: Any) -> str:
        """
        Coerce model-produced final validation into schema-compliant Validate line.
        """

        return ScriptExporter.__normalize_validation_line(
            value=value,
            fallback="Validate expected goal state is visible.",
        )

    @staticmethod
    def __normalize_validation_line(value: Any, *, fallback: str) -> str:
        """
        Coerce arbitrary text into a schema-compliant Validate line.
        """

        raw = str(value or "").strip()
        if not raw:
            return fallback

        match = re.search(pattern=r"\bvalidate\b.*", string=raw, flags=re.IGNORECASE)
        if match:
            extracted = match.group(0).strip()
            return "Validate" + extracted[len("validate") :] if extracted else fallback

        cleaned = raw.rstrip(".")
        if cleaned.lower().startswith("that "):
            return f"Validate {cleaned}."
        return f"Validate that {cleaned[0].lower() + cleaned[1:] if len(cleaned) > 1 else cleaned.lower()}."

    @staticmethod
    def __intent_requires_if_block(intent: str) -> bool:
        """
        Determine whether intent explicitly requests conditional flow.
        """

        normalized = ScriptExporter.__normalize_text_signal(text=intent)
        if not normalized:
            return False

        # Previously this was restricted to cart/clear-cart flows; that was too narrow
        # and missed general user-demanded conditionals like "If the dropdown is visible".
        conditional_terms = (" if ", " when ", " if_", " if-", " if(", " if the", " if there")
        return any(term in f" {normalized} " for term in conditional_terms)

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
            return None

        if not self.__prompt_builder:
            raise ScriptExportError("Script exporter prompt builder is not configured.")

        # Use Gemini to robustly extract validation subjects from the intent
        validation_subjects = await self.__extract_validation_subjects_with_llm(
            intent=(intent or goal_state)
        )
        if validation_subjects:
            logger.info(
                f"Using Gemini-extracted validation subjects ({len(validation_subjects)} found) "
                "for LLM export baseline generation."
            )

        llm_baseline_script = self.export(
            step_results=step_results,
            intent=intent,
            goal_state=goal_state,
            package_name=package_name,
            include_final_validation=False,
            validation_subjects_override=validation_subjects,
        )
        deterministic_fallback_script = self.export(
            step_results=step_results,
            intent=intent,
            goal_state=goal_state,
            package_name=package_name,
            include_final_validation=True,
            validation_subjects_override=validation_subjects,
        )
        deterministic_fallback_script = ScriptExporter.__normalize_script_output(
            script=deterministic_fallback_script
        )
        deterministic_fallback_script = ScriptExporter.__sanitize_script_targets(
            script=deterministic_fallback_script, intent=(intent or goal_state)
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

        # Guardrail (soft): record when catalog lines still contain generic targets.
        # We no longer hard-reject structured export here, because that prevented
        # legitimate conditional blocks from being emitted when the user explicitly
        # requested them. Instead we log for observability and rely on downstream
        # sanitization and ScriptExportPayload validation to enforce safety.
        for action_id, line in action_catalog.items():
            if line.strip().lower().startswith("open_app "):
                continue
            target_phrase = ScriptExporter.__extract_target_from_action_line(line=line) or ""
            if Normalizer.is_generic_target_name(target_phrase):
                logger.warning(
                    "Gemini structured export catalog contains generic target name "
                    "[export_violation=generic_target_name, action_id=%s, line=%s].",
                    action_id,
                    line,
                )

        action_validation_baseline_script = ScriptExporter.__normalize_script_output(
            script="\n".join(action_catalog.values())
        )
        action_validation_baseline_script = ScriptExporter.__sanitize_script_targets(
            script=action_validation_baseline_script,
            intent=(intent or goal_state),
        )
        if not action_validation_baseline_script.strip():
            action_validation_baseline_script = llm_baseline_script
        require_if_block = ScriptExporter.__intent_requires_if_block(intent=(intent or goal_state))
        required_open_app_line = (
            str(action_catalog.get(required_open_app_id) or "").strip().lower()
            if required_open_app_id
            else None
        )

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
            logger.warning(
                "Gemini emit_script arguments missing; "
                "falling back to deterministic exporter output "
                "[export_mode=fallback_deterministic, export_violation=missing_tool_args]."
            )
            return deterministic_fallback_script

        try:
            raw_emit_args = EmitScriptArgs.model_validate(structured_args)
        except Exception as exception:
            logger.warning(
                "Gemini emit_script raw payload parsing failed (%s); "
                "falling back to deterministic exporter output "
                "[export_mode=invalid_structured, export_violation=raw_parse].",
                exception,
            )
            return deterministic_fallback_script

        normalized_structured_args = ScriptExporter.__normalize_structured_action_ids(
            structured_args=raw_emit_args.model_dump(exclude_unset=True),
            required_action_ids=required_action_ids,
            action_catalog=action_catalog,
        )

        try:
            structured_payload = ScriptExportStructuredPayload.model_validate(
                {
                    **normalized_structured_args,
                    "action_catalog": action_catalog,
                    "required_action_ids": required_action_ids,
                    "required_open_app_id": required_open_app_id,
                    "require_if_block": require_if_block,
                    "expected_validation_count": len(validation_subjects),
                }
            )
        except Exception as exception:
            logger.warning(
                "Gemini structured payload validation failed (%s); "
                "falling back to deterministic exporter output "
                "[export_mode=fallback_deterministic, export_violation=structured_payload].",
                exception,
            )
            return deterministic_fallback_script

        raw_structured_script = structured_payload.to_script()
        candidate = ScriptExporter.__normalize_script_output(script=raw_structured_script)
        candidate = ScriptExporter.__sanitize_script_targets(
            script=candidate, intent=(intent or goal_state)
        )

        try:
            parsed_script = ScriptExportPayload.model_validate(
                {
                    "script": candidate,
                    "allowed_action_lines": allowed_action_lines,
                    "required_open_app": required_open_app_line,
                    "require_if_block": require_if_block,
                }
            )
        except Exception as exception:
            logger.warning(
                "Gemini script schema validation failed (%s); "
                "falling back to deterministic exporter output "
                "[export_mode=fallback_deterministic, export_violation=script_schema].",
                exception,
            )
            return deterministic_fallback_script

        candidate = parsed_script.script
        if not ScriptExporter.__is_valid_llm_script(
            candidate=candidate, baseline=action_validation_baseline_script
        ):
            logger.warning(
                "Gemini script failed structural/action coverage validation; "
                "falling back to deterministic exporter output "
                "[export_mode=fallback_deterministic, export_violation=post_validation_coverage]."
            )
            return deterministic_fallback_script
        if not ScriptExporter.__contains_goal_validation(script=candidate):
            logger.warning(
                "Gemini script missing final goal validation; "
                "falling back to deterministic exporter output "
                "[export_mode=fallback_deterministic, export_violation=missing_goal_validation]."
            )
            return deterministic_fallback_script

        logger.info(
            "Gemini script export succeeded via structured payload [export_mode=llm_structured]."
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

        n = len(step_results)
        i = 0
        while i < n:
            step = step_results[i]
            if ScriptExporter.__is_launcher_activity(activity=ScriptExporter.__get_activity(step)):
                i += 1
                continue
            action_type_val = ScriptExporter.__get_action_type(step=step)
            target = ScriptExporter.__resolve_target(step=step)

            if isinstance(step, StepResult):
                text = step.step.action.text
                rationale = step.step.action.rationale
                wait_duration = step.step.action.wait_duration
                is_app_launcher_signal = step.step.action.is_app_launcher
                wait_subject = step.step.action.wait_subject
                wait_pattern = step.step.action.wait_pattern
                scroll_target = step.step.action.scroll_target
            else:
                text = step.get("text")
                rationale = str(object=step.get("rationale", ""))
                wait_duration = step.get("wait_duration")
                is_app_launcher_signal = step.get("is_app_launcher", False)
                wait_subject = step.get("wait_subject")
                wait_pattern = step.get("wait_pattern")
                scroll_target = step.get("scroll_target")

            if action_type_val == "wait" and target.lower() in ScriptExporter.__GENERIC_TARGETS:
                # Prefer structured wait_subject field with pattern as fallback to rationale
                if wait_subject:
                    target = wait_subject
                elif wait_pattern:
                    pattern_map = {
                        "ad": "ad to finish",
                        "splash": "app to finish loading",
                        "load": "app to finish loading",
                        "search": "search results to appear",
                    }
                    target = pattern_map.get(wait_pattern, "screen to load")
                else:
                    target = ScriptExporter.__infer_wait_subject(
                        rationale=rationale, wait_subject=None
                    )

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
                    # Prefer structured scroll_target, fallback to inference
                    next_target = (
                        scroll_target
                        or ScriptExporter.__infer_scroll_target(
                            steps=step_results, start=swipe_start, end=i
                        )
                        or ScriptExporter.__extract_goal_label(goal_state=intent)
                        or intent
                        or "the target"
                    )

                label = ScriptExporter.__swipe_direction_label(action_type=swipe_direction)
                lines.append(f"{label} until {next_target} is visible")
                continue

            description = Normalizer.action(
                action_type=action_type_val, target=target, text=text, wait_duration=wait_duration
            )

            lowered = description.lower()
            if lowered.startswith("validate "):
                i += 1
                continue

            is_first_step_derived_action = (
                bool(package_name) and len(lines) == 1 and lines[0].lower().startswith("open_app ")
            )
            # Collapse app launch into OPEN_APP by skipping the initial app-icon tap.
            # Match if: explicit is_app_launcher signal OR heuristic pattern match
            is_launch_tap = (
                is_first_step_derived_action
                and action_type_val == "tap"
                and (
                    is_app_launcher_signal
                    or ScriptExporter.__is_likely_launch_tap(
                        target=target,
                        description=description,
                    )
                )
            )
            if is_launch_tap:
                logger.debug(
                    f"[EXPORTER] Collapsing launcher tap into OPEN_APP: target='{target}' "
                    f"description='{description}' package={package_name} "
                    f"launcher_signal={is_app_launcher_signal}"
                )
                i += 1
                continue

            lines.append(description)
            i += 1

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
    def __is_likely_launch_tap(target: str, description: str) -> bool:
        """
        Heuristic for identifying the launcher tap that opens the app.
        Matches patterns like "Chrome icon", "app icon", "<name> icon", "launcher icon", etc.
        More conservative to avoid false positives on element names that happen to end in 'icon'.
        """

        combined = f"{target} {description}".strip().lower()
        if not combined:
            return False

        # Most reliable: explicit "app icon" or "launcher icon" phrase
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

        # Conservative pattern: word characters followed by "icon" as a distinct token
        # This matches "Chrome icon", "Maps icon", "1mg icon" but avoids false
        # positives on random text containing the substring "icon"
        if re.search(r"\b(?:the\s+)?[a-z0-9.\-_'\s]+\s+icon\b", combined):
            return True

        # Fallback: ends with " icon" (catches pattern-matched targets)
        return combined.endswith(" icon")

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
    def __infer_target_from_rationale(
        *, action_type: str, rationale: Optional[str], fallback: str
    ) -> str:
        """
        Recover a human-readable target from rationale when model target is generic.
        """

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
            if name and name.lower() not in ScriptExporter.__GENERIC_TARGETS:
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
                and lowered not in ScriptExporter.__GENERIC_TARGETS
                and lowered not in ("app", "application", "screen")
            ):
                return candidate

        return fallback

    @staticmethod
    def __should_generalize_target(rationale: Optional[str]) -> bool:
        """
        Detect if rationale indicates non-specific selection requiring generalization.
        """

        if not rationale:
            return False

        text = str(rationale).lower()

        # Match patterns like "first item", "any item", "random category", etc.
        pattern = r"\b(first|any|random|any available|available)\s+(item|product|option|category|choice|element)\b"
        return bool(re.search(pattern=pattern, string=text, flags=re.IGNORECASE))

    @staticmethod
    def __generalize_product_target(target: str, rationale: Optional[str]) -> str:
        """
        Extract generic action from product-specific targets.

        Examples:
            "ADD button for Limcee Vitamin C Chewable Tablet" -> "ADD button"
            "Remove button for Product X" -> "Remove button"

        Args:
            target: The target description to generalize
            rationale: The reasoning context for target selection
        """

        if not target:
            return target

        cleaned = Normalizer.clean(text=target)
        if not cleaned:
            return target

        # Use rationale to detect element type hints
        rationale_lower = str(rationale).lower() if rationale else ""
        detected_element_type = None

        if "button" in rationale_lower:
            detected_element_type = "button"
        elif "icon" in rationale_lower:
            detected_element_type = "icon"
        elif "option" in rationale_lower:
            detected_element_type = "option"

        # Pattern: "<action> button for <product>"
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

        # If no clear pattern, return original
        return target

    @staticmethod
    def __resolve_target(step: Union[StepResult, Dict[str, Any]]) -> str:
        """
        Resolve the description for a target.
        """

        rationale: Optional[str]
        script_target: Optional[str] = None
        scroll_target: Optional[str] = None
        wait_subject: Optional[str] = None

        if isinstance(step, StepResult):
            if step.generalized_target:
                return ScriptExporter.__normalize_positional(target=step.generalized_target)
            action = step.step.action
            target = action.natural_language_target or action.target
            rationale = action.rationale
            script_target = action.script_target
            scroll_target = action.scroll_target
            wait_subject = action.wait_subject
        else:
            if step.get("generalized_target"):
                raw = str(object=step.get("generalized_target") or "")
                return ScriptExporter.__normalize_positional(target=raw)
            target = step.get("natural_language_target") or step.get("target") or ""
            rationale = str(object=step.get("rationale") or "")
            script_target = step.get("script_target")
            scroll_target = step.get("scroll_target")
            wait_subject = step.get("wait_subject")

        resolved_target = Normalizer.clean(text=target) or "element"
        lower_resolved = resolved_target.lower()

        # When the model emitted a generic target, prefer structured fields first.
        if lower_resolved in ScriptExporter.__GENERIC_TARGETS:
            for candidate in (script_target, scroll_target, wait_subject):
                if candidate and not Normalizer.is_generic_target_name(candidate):
                    return Normalizer.clean(text=candidate)

            action_type = ScriptExporter.__get_action_type(step=step)
            inferred = ScriptExporter.__infer_target_from_rationale(
                action_type=action_type,
                rationale=rationale,
                fallback=resolved_target,
            )
            if inferred and not Normalizer.is_generic_target_name(inferred):
                return inferred

            # As a last resort, keep a simple, generic 'element' label rather than
            # exposing internal IDs or coordinates.
            return "element"

        # Generalize product-specific targets when rationale indicates non-specific selection
        if ScriptExporter.__should_generalize_target(rationale=rationale):
            generalized = ScriptExporter.__generalize_product_target(
                target=resolved_target,
                rationale=rationale,
            )
            if generalized != resolved_target:
                return generalized

        return resolved_target

    @staticmethod
    def __extract_target_from_action_line(line: str) -> Optional[str]:
        """
        Best-effort extraction of the target phrase from an executable script line.
        """

        text = line.strip()
        lower = text.lower()

        if lower.startswith("tap on "):
            return text[len("Tap on ") :].strip()

        if lower.startswith("type "):
            marker = lower.rfind(" into ")
            if marker != -1:
                return text[marker + len(" into ") :].strip()

        if lower.startswith("scroll until you see "):
            return text[len("Scroll until you see ") :].strip()
        if lower.startswith("scroll down until ") or lower.startswith("scroll up until "):
            suffix = (
                text[len("Scroll down until ") :]
                if lower.startswith("scroll down until ")
                else text[len("Scroll up until ") :]
            )
            return suffix.strip()

        if lower.startswith("wait for "):
            return text[len("Wait for ") :].strip()

        if lower.startswith("long press on "):
            return text[len("Long press on ") :].strip()

        if lower.startswith("validate "):
            target = text[len("Validate ") :].strip()
            if target.lower().startswith("that "):
                target = target[5:].strip()
            return target

        # OPEN_APP and other non-target lines: no target phrase to check for generic.
        if lower.startswith("open_app "):
            return None
        return None

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
    def __is_launcher_activity(activity: str) -> bool:
        """
        Return whether the activity belongs to a launcher package.
        """

        text = str(activity or "").strip()
        if not text:
            return False

        package = text.split("/")[0]
        return package in LAUNCHER_PACKAGES

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
        validation_subjects_override: Optional[Sequence[str]] = None,
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
        if validation_subjects_override is not None:
            validation_subjects = []
            for subject in validation_subjects_override:
                cleaned_subject = Normalizer.clean(text=str(subject).strip(" .,:;"))
                if cleaned_subject:
                    validation_subjects.append(cleaned_subject)
        else:
            validation_subjects = ScriptExporter.__extract_validation_subjects(
                intent=(intent or goal_state)
            )
        validation_subject_index = 0
        reserved_final_subjects = 1 if include_final_validation and validation_subjects else 0
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
                    # Keep the first recorded step when no launch boundary is found.
                    # Skipping it can drop required actions from exported scripts.
                    i = 0

        while i < n:
            step = step_results[i]
            if ScriptExporter.__is_launcher_activity(activity=ScriptExporter.__get_activity(step)):
                i += 1
                continue
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
            deferred_screen_validation: Optional[str] = None
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
                    available_for_intermediate = max(
                        0, len(validation_subjects) - reserved_final_subjects
                    )
                    if validation_subject_index < available_for_intermediate:
                        requested_subject = validation_subjects[validation_subject_index]
                        validation_subject_index += 1
                        val_line = Normalizer.validation(target=requested_subject, explicit=True)
                    else:
                        val_line = f"Validate {target} is visible"
                    if prev_condition:
                        deferred_screen_validation = f"IF {prev_condition} {{ {val_line} }}"
                    else:
                        deferred_screen_validation = val_line

            swipe_just_processed = False

            if isinstance(step, StepResult):
                action = step.step.action
                text = action.text
                rationale = action.rationale
                wait_duration = action.wait_duration
                is_app_launcher_signal = action.is_app_launcher
                wait_subject = action.wait_subject
                wait_pattern = action.wait_pattern
            else:
                text = step.get("text")
                rationale = str(object=step.get("rationale", ""))
                wait_duration = step.get("wait_duration")
                is_app_launcher_signal = bool(step.get("is_app_launcher", False))
                wait_subject = step.get("wait_subject")
                wait_pattern = step.get("wait_pattern")

            if action_type_val == "wait" and target.lower() in ScriptExporter.__GENERIC_TARGETS:
                # Prefer structured wait_subject field with pattern as fallback to rationale
                if wait_subject:
                    target = wait_subject
                elif wait_pattern:
                    pattern_map = {
                        "ad": "ad to finish",
                        "splash": "app to finish loading",
                        "load": "app to finish loading",
                        "search": "search results to appear",
                    }
                    target = pattern_map.get(wait_pattern, "screen to load")
                else:
                    target = ScriptExporter.__infer_wait_subject(
                        rationale=rationale, wait_subject=None
                    )

            description = Normalizer.action(
                action_type=action_type_val, target=target, text=text, wait_duration=wait_duration
            )

            is_first_step_derived_action = (
                bool(package_name) and len(lines) == 1 and lines[0].lower().startswith("open_app ")
            )
            is_launch_tap = (
                is_first_step_derived_action
                and action_type_val == "tap"
                and (
                    is_app_launcher_signal
                    or ScriptExporter.__is_likely_launch_tap(
                        target=target,
                        description=description,
                    )
                )
            )
            if is_launch_tap:
                i += 1
                continue

            # Semantic Validation Handling (Integrated)
            if event_type == "validation":
                effective_target = target
                is_system_validation = ScriptExporter.__is_system_validation(
                    target=target, rationale=rationale, condition=condition
                )
                should_use_intent_subject = (
                    bool(validation_subjects)
                    and not is_system_validation
                    and validation_subject_index
                    < max(0, len(validation_subjects) - reserved_final_subjects)
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

            # Emit smart screen-change validation after the associated action.
            if deferred_screen_validation and (
                not lines or lines[-1].strip() != deferred_screen_validation
            ):
                lines.append(deferred_screen_validation)
            i += 1

        # Final Goal Validation Logic (Restored)
        if include_final_validation and step_results:
            last_action_type = ScriptExporter.__get_action_type(step=step_results[-1])

            if last_action_type not in ("complete", "verify_goal_completion"):
                if validation_subject_index < len(validation_subjects):
                    final_subject = validation_subjects[validation_subject_index]
                    final_validation_line = f"Validate that {final_subject}"
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
    def __extract_validation_subjects_regex(intent: str) -> list[str]:
        """
        Extract all user-requested validation subjects from intent text using regex.
        This is a fallback when LLM-based extraction is unavailable.
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

    async def __extract_validation_subjects_with_llm(self, intent: str) -> list[str]:
        """
        Extract validation subjects robustly using Gemini NLP understanding.
        Handles complex natural language like numbered lists, conditional validations, etc.
        Falls back to regex if LLM fails or is unavailable.
        """

        if not intent or not self.__llm:
            return ScriptExporter.__extract_validation_subjects_regex(intent=intent)

        try:
            system_instruction = (
                "You are an expert at parsing user intents for mobile UI automation. "
                "Your task is to extract all validation requirements from a user's intent."
            )

            prompt = (
                f"Extract all validation requirements from this intent. "
                f"Return a JSON list of validation subjects (what to validate/confirm/check). "
                f"Each subject should be a complete, standalone assertion (e.g., 'the cart page is displayed', 'api validation succeeded'). "
                f"Handle numbered lists, conditionals, and complex sentences. Do not include keywords like 'Validate' or 'Check'. "
                f"Return ONLY valid JSON list of strings, no other text.\n\n"
                f"Intent: {intent}"
            )

            response = await self.__llm.generate(
                use_cache=False,
                prompt=[prompt],
                system_instruction=system_instruction,
            )

            if not response or not response.content:
                logger.warning(
                    "Gemini validation subject extraction returned empty response; using regex fallback."
                )
                return ScriptExporter.__extract_validation_subjects_regex(intent=intent)

            # Parse JSON response
            try:
                # Clean response (remove markdown code blocks if present)
                content = response.content.strip()
                if content.startswith("```json"):
                    content = content[7:]
                if content.startswith("```"):
                    content = content[3:]
                if content.endswith("```"):
                    content = content[:-3]
                content = content.strip()

                subjects = json.loads(content)
                if not isinstance(subjects, list):
                    logger.warning("Gemini returned non-list JSON; using regex fallback.")
                    return ScriptExporter.__extract_validation_subjects_regex(intent=intent)

                # Normalize subjects
                normalized = []
                for subject in subjects:
                    if isinstance(subject, str):
                        cleaned = Normalizer.clean(text=subject.strip())
                        if cleaned:
                            normalized.append(cleaned)

                if normalized:
                    logger.info(
                        f"Gemini extracted {len(normalized)} validation subjects from intent."
                    )
                    return normalized
                else:
                    logger.warning("Gemini extracted empty subjects; using regex fallback.")
                    return ScriptExporter.__extract_validation_subjects_regex(intent=intent)

            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse Gemini JSON response ({e}); using regex fallback.")
                return ScriptExporter.__extract_validation_subjects_regex(intent=intent)

        except Exception as e:
            logger.warning(
                f"Gemini validation subject extraction failed ({e}); falling back to regex extraction."
            )
            return ScriptExporter.__extract_validation_subjects_regex(intent=intent)

    @staticmethod
    def __extract_validation_subjects(intent: str) -> list[str]:
        """
        Extract validation subjects using regex (synchronous fallback).
        For async LLM-based extraction, use __extract_validation_subjects_with_llm().
        """

        return ScriptExporter.__extract_validation_subjects_regex(intent=intent)

    @staticmethod
    def __infer_scroll_target(
        steps: Sequence[Union[StepResult, Dict[str, Any]]],
        start: int,
        end: int,
    ) -> str:
        """
        Infer what the user was scrolling to find from structured scroll_target field or swipe rationales.
        Prioritizes structured scroll_target over rationale parsing.
        """

        # First check if any step has structured scroll_target
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

        # Fallback to rationale parsing
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
            clause = verb_match.group(1).strip()

            # Try extracting quoted phrases first (e.g., 'Vitamins and supplements', "Lab tests")
            quoted_match = re.search(r"['\"]([^'\"]+)['\"]", clause)
            if quoted_match:
                extracted = quoted_match.group(1).strip()
                # Clean up common suffixes
                extracted = re.sub(
                    r"\s+(section|category|area|page|screen|button|tab|widget)$",
                    "",
                    extracted,
                    flags=re.IGNORECASE,
                )
                if extracted:
                    return extracted

            # Fall back to proper phrase matching
            product_match = ScriptExporter.__PROPER_PHRASE_RE.search(string=clause)
            if product_match:
                return product_match.group(1).strip()

            # Last resort: clean up the clause and extract meaningful content
            # Remove common stop words and suffixes
            cleaned = re.sub(r"^the\s+", "", clause, flags=re.IGNORECASE)
            cleaned = re.sub(
                r"\s+(section|category|area|page|screen|button|tab|widget)$",
                "",
                cleaned,
                flags=re.IGNORECASE,
            )
            cleaned = cleaned.strip(" ,'\"")
            if cleaned and len(cleaned) > 5:  # Avoid too-short generic phrases
                return cleaned
        return ""

    @staticmethod
    def __infer_wait_subject(rationale: Optional[str], wait_subject: Optional[str] = None) -> str:
        """
        Derive a human-readable wait subject from structured wait_subject field or step rationale.
        Prioritizes structured wait_subject over rationale parsing.
        """

        # Use structured field if available
        if wait_subject:
            return wait_subject

        # Fallback to rationale normalization
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
