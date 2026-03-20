from __future__ import annotations

from logging import getLogger
from typing import Any, Dict, Optional, Sequence, Union

from fathom.core.exceptions import ScriptExportError
from fathom.core.prompts.export import ExportPromptBuilder
from fathom.core.prompts.factory import PromptFactory
from fathom.core.prompts.tools import ToolRegistry
from fathom.core.services.exporter.action_catalog import build_action_catalog_from_steps
from fathom.core.services.exporter.deterministic_export import export_steps_to_script
from fathom.core.services.exporter.script_text import (
    extract_target_from_action_line,
    intent_requires_if_block,
    normalize_script_output,
    sanitize_script_targets,
)
from fathom.core.services.exporter.structured_export import (
    contains_goal_validation,
    is_valid_llm_script,
    normalize_structured_action_ids,
)
from fathom.core.services.exporter.trace_payload import build_export_payload
from fathom.core.services.exporter.validation_subjects import extract_validation_subjects_with_llm
from fathom.core.services.normalizer import Normalizer
from fathom.interfaces.llm import LLMPort
from fathom.schemas.export import (
    ScriptExportPayload,
    ScriptExportStructuredPayload,
    ScriptExportStructuredPayloadShape,
)
from fathom.schemas.gemini_tools import EmitScriptArgs
from fathom.schemas.steps import StepResult

logger = getLogger(__name__)


class ScriptExporter:
    """
    Service for exporting execution history to natural language scripts.
    """

    def __init__(self, *, llm: Optional[LLMPort] = None, use_cache: bool = True) -> None:
        self.__llm = llm
        self.__use_cache = use_cache
        self.__prompt_builder: Optional[ExportPromptBuilder] = (
            PromptFactory.get_export_builder(model_name=llm.model_name) if llm else None
        )

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
        return export_steps_to_script(
            step_results,
            intent=intent,
            goal_state=goal_state,
            package_name=package_name,
            include_final_validation=include_final_validation,
            validation_subjects_override=validation_subjects_override,
        )

    async def export_with_llm(
        self,
        step_results: Sequence[Union[StepResult, Dict[str, Any]]],
        *,
        intent: str = "",
        goal_state: str = "",
        package_name: str = "",
    ) -> Optional[str]:
        if not self.__llm:
            return None

        payload = build_export_payload(step_results=step_results)
        if not payload:
            return None

        if not self.__prompt_builder:
            raise ScriptExportError("Script exporter prompt builder is not configured.")

        validation_subjects = await extract_validation_subjects_with_llm(
            llm=self.__llm, intent=(intent or goal_state)
        )
        if validation_subjects:
            logger.info(
                f"Using Gemini-extracted validation subjects ({len(validation_subjects)} found) "
                "for LLM export baseline generation."
            )

        llm_baseline_script = ScriptExporter.export(
            step_results=step_results,
            intent=intent,
            goal_state=goal_state,
            package_name=package_name,
            include_final_validation=False,
            validation_subjects_override=validation_subjects,
        )
        deterministic_fallback_script = ScriptExporter.export(
            step_results=step_results,
            intent=intent,
            goal_state=goal_state,
            package_name=package_name,
            include_final_validation=True,
            validation_subjects_override=validation_subjects,
        )
        deterministic_fallback_script = normalize_script_output(
            script=deterministic_fallback_script
        )
        deterministic_fallback_script = sanitize_script_targets(
            script=deterministic_fallback_script, intent=(intent or goal_state)
        )
        llm_baseline_script = normalize_script_output(script=llm_baseline_script)
        llm_baseline_script = sanitize_script_targets(
            script=llm_baseline_script, intent=(intent or goal_state)
        )
        action_catalog, required_action_ids, required_open_app_id = build_action_catalog_from_steps(
            step_results=step_results,
            package_name=package_name,
            intent=(intent or goal_state),
        )
        action_catalog_lines = [
            f"{action_id}: {line}" for action_id, line in action_catalog.items()
        ]
        allowed_action_lines = [
            line.strip().lower() for line in action_catalog.values() if line.strip()
        ]

        for action_id, line in action_catalog.items():
            if line.strip().lower().startswith("open_app "):
                continue
            target_phrase = extract_target_from_action_line(line=line) or ""
            if Normalizer.is_generic_target_name(target_phrase):
                logger.warning(
                    "Gemini structured export catalog contains generic target name "
                    "[export_violation=generic_target_name, action_id=%s, line=%s].",
                    action_id,
                    line,
                )

        action_validation_baseline_script = normalize_script_output(
            script="\n".join(action_catalog.values())
        )
        action_validation_baseline_script = sanitize_script_targets(
            script=action_validation_baseline_script,
            intent=(intent or goal_state),
        )
        if not action_validation_baseline_script.strip():
            action_validation_baseline_script = llm_baseline_script
        require_if_block = intent_requires_if_block(intent=(intent or goal_state))
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

        normalized_structured_args = normalize_structured_action_ids(
            structured_args=raw_emit_args.model_dump(exclude_unset=True),
            required_action_ids=required_action_ids,
            action_catalog=action_catalog,
        )

        try:
            shape = ScriptExportStructuredPayloadShape.model_validate(normalized_structured_args)
            structured_payload = ScriptExportStructuredPayload.enforce_policy(
                shape=shape,
                action_catalog=action_catalog,
                required_action_ids=required_action_ids,
                required_open_app_id=required_open_app_id,
                require_if_block=require_if_block,
                expected_validation_count=len(validation_subjects),
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
        candidate = normalize_script_output(script=raw_structured_script)
        candidate = sanitize_script_targets(script=candidate, intent=(intent or goal_state))

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
        if not is_valid_llm_script(candidate=candidate, baseline=action_validation_baseline_script):
            logger.warning(
                "Gemini script failed structural/action coverage validation; "
                "falling back to deterministic exporter output "
                "[export_mode=fallback_deterministic, export_violation=post_validation_coverage]."
            )
            return deterministic_fallback_script
        if not contains_goal_validation(script=candidate):
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
