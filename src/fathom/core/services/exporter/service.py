from __future__ import annotations

from logging import getLogger
from typing import Any, Dict, Optional, Sequence, Union

from fathom.core.exceptions import ScriptExportError
from fathom.core.prompts.export import ExportPromptBuilder
from fathom.core.prompts.factory import PromptFactory
from fathom.core.prompts.tools import ToolRegistry
from fathom.core.services.exporter.action_catalog import build_action_catalog_from_steps
from fathom.core.services.exporter.script_text import normalize_script_output
from fathom.core.services.exporter.structured_export import (
    contains_goal_validation,
    is_valid_llm_script,
)
from fathom.core.services.exporter.trace_payload import build_export_payload
from fathom.core.services.exporter.validation_subjects import extract_validation_subjects_with_llm
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

        require_if_block = False
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
            raise ScriptExportError(
                "Gemini emit_script arguments missing [export_violation=missing_tool_args]."
            )

        try:
            raw_emit_args = EmitScriptArgs.model_validate(structured_args)
        except Exception as exception:
            raise ScriptExportError(
                f"Gemini emit_script raw payload parsing failed: {exception} "
                "[export_violation=raw_parse]."
            ) from exception

        try:
            shape = ScriptExportStructuredPayloadShape.model_validate(
                raw_emit_args.model_dump(exclude_unset=True)
            )
            structured_payload = ScriptExportStructuredPayload.enforce_policy(
                shape=shape,
                action_catalog=action_catalog,
                required_action_ids=required_action_ids,
                required_open_app_id=required_open_app_id,
                require_if_block=require_if_block,
                expected_validation_count=len(validation_subjects),
            )
        except Exception as exception:
            raise ScriptExportError(
                f"Gemini structured payload validation failed: {exception} "
                "[export_violation=structured_payload]."
            ) from exception

        raw_structured_script = structured_payload.to_script()
        candidate = normalize_script_output(script=raw_structured_script)

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
            raise ScriptExportError(
                f"Gemini script schema validation failed: {exception} "
                "[export_violation=script_schema]."
            ) from exception

        candidate = parsed_script.script
        if not is_valid_llm_script(candidate=candidate, catalog_action_count=len(action_catalog)):
            raise ScriptExportError(
                "Gemini script failed structural/action coverage validation "
                "[export_violation=post_validation_coverage]."
            )
        if not contains_goal_validation(script=candidate):
            raise ScriptExportError(
                "Gemini script missing final goal validation "
                "[export_violation=missing_goal_validation]."
            )

        logger.info(
            "Gemini script export succeeded via structured payload [export_mode=llm_structured]."
        )
        return candidate
